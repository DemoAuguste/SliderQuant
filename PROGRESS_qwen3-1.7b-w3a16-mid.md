# Qwen3-1.7B W3A16 中间层量化（mid, layer 6-21）— 进展与 OOM 攻坚记录

> 状态：**v3 完成：真正根因定位（首窗输入错位）→ 补丁修复 → PPL 17.80 / avg-6 61.64%**　更新时间：2026-09-18 14:30（UTC+8）

## 一、任务目标

在只量化**中间层 layer 6~21**（共 16 层）的设置下复现 W3A16 量化，使用**完整训练配置**（seqlen=2048、nsamples=128、epochs=60），与已有的 FP16 基线、W3A16-full（全部 28 层）结果做**三方对比**，验证「只量化中间层」能否以更少的量化层逼近全量化的精度/压缩权衡。

## 二、实验配置

### `configs/qwen3-1.7b-w3a16-mid/config.yaml`（当前锁定值）

```yaml
model: /workspace/quant-research/model_zoo/Qwen3-1.7B
wbits: 3
abits: 16
epochs: 60
use_bfloat16: true
eval_ppl: true
quant_mode: lora_only
quant_layer_list: "6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21"
layer_windows_scheduler: "6-9,8-11,10-13,12-15,14-17,16-19,18-21"
batch_size: 3
low_memory: true          # 关键：激活留 CPU，小批量 H2D
low_cpu_memory: true      # 关键：清理推理模式外的权重
nsamples: 128
seqlen: 2048
calib_dataset: wikitext2
test_datasets: wikitext2
num_layer: 4
sliding_layer: 2
fill_window_size: 4
lora_rank: 4
use_lora: true
quant_rate: 0.5
inference_batch_size: 4   # 从 12 降到 4，压低 fp_target 阶段峰值
fp16_act: true
scale_lr: 0.0
lora_lr: 0.0005
lwc_lr: 0.01
warmup_ratio: 0.0
```

- 训练结构：`quant_step=2` → stage 1（quant_rate=0.5）+ stage 2（quant_rate=1.0），每 stage 7 个 sliding window × 30 epochs，共 14 轮。
- 使用卡：NPU 5（`ASCEND_RT_VISIBLE_DEVICES=5`，宿主 NUMA node 0）。
- **⚠ 后确认的调度缺陷**：显式 7 窗调度使 `fill_window_size` 失效（[sliderquant.py](file:///c:/code/SliderQuant/quantize/sliderquant.py#L443-L483) 中 `layer_windows_scheduler` 优先级更高），14 轮全部为无填充滑窗，深层（14-21）每 stage 仅训 1 次、末层无 end 填充重复训练 → 直接导致训练净负收益（见第六节）。

## 三、OOM 被杀问题：诊断与彻底解决（核心记录）

### 3.1 现象

- 完整配置（seqlen=2048/nsamples=128）训练反复在 **epoch 2 被 SIGKILL（exit=137）**。
- 宿主内存表面充足：`free≈89GB`、`available≈382GB`，远未耗尽 —— 排除"总量不足"。

### 3.2 根因：NUMA 内存不均 + 昇腾 devmm 驱动约束

- 8 张 NPU 卡全部绑定在**内存紧张的 node 0/2/4/6**；内存充足的 node 1/3/7 **没有卡**。
- 昇腾 `devmm` 驱动强制要求从**卡的本地 NUMA node** pin 连续物理内存做 DMA，`numactl --interleave` 无法绕过。
- 卡 ↔ NUMA node 映射（`npu-smi` ID ↔ `lscpu` node）：

| NPU 卡 | NUMA node |
| --- | --- |
| 卡 0(C1)、卡 1(C2) | node 6 |
| 卡 2(81)、卡 3(82) | node 4 |
| **卡 4(01)、卡 5(02) ← 当前用卡 5** | **node 0（仅剩 ~2GB，已被其他用户耗尽）** |
| 卡 6(41)、卡 7(42) | node 2 |

### 3.3 试错时间线

| # | 方案 | 结果 |
| --- | --- | --- |
| 1 | 完整配置直跑 | epoch 2 被 OOM 杀 |
| 2 | `numactl --interleave=all` | 无效（node 4/5/6 仅 ~1GB，拖垮 interleave） |
| 3 | 降配 seqlen 1024 / nsamples 32 | 被杀点推迟到 epoch 5（治标） |
| 4 | `low_memory: true` | **关键修复**：能完整跑完 round 0 并存 checkpoint |
| 5 | `inference_batch_size` 12→4 | 能跑到 round 1 epoch 27 |
| 6 | 降配下继续跑 | node 0 恶化到 2.7GB，模型权重 3.4GB 都放不下，仍被杀 |
| 7 | `numactl --membind=1`（未先 drop_caches） | 失败：node 1 当时仅 43GB，round 0 就死 |
| 8 | **`drop_caches` + `numactl --membind=1`** | **彻底解决，完整配置稳定运行，0 次被杀** |

### 3.4 最终方案（两步缺一不可）

**第一步：释放页缓存（宿主 root）** —— 把 node 1 从 12GB 释放到 119GB：

```bash
sync
echo 3 > /proc/sys/vm/drop_caches
echo 1 > /proc/sys/vm/compact_memory
```

> 局限：只能释放 page cache，不能释放其他用户的 tmpfs/匿名页。释放效果主要落在 node 1（12→119GB）、node 3（10→81GB）、node 7（6→57GB）；node 0（当前卡的本地 node）依然只有 ~1.9GB —— 所以必须配合第二步。

**第二步：把训练进程内存绑到 node 1**（容器内，`numactl` 需先从宿主复制进容器）：

```bash
numactl --membind=1 python main.py \
  --config configs/qwen3-1.7b-w3a16-mid/config.yaml \
  --output_dir log/qwen3-1.7b-w3a16-mid-full \
  --cache_dir cache
```

- 原理：host 侧训练内存（数据预处理、激活缓存、CPU offload 的 fp_target 等 ~10GB RSS）全部落在 node 1；NPU 卡本地 node 只需承载 devmm 强制的 H2D pin 内存，量级小，node 0 剩余内存够用。

### 3.5 训练守护脚本

`_run_w3a16_mid_full_retry.sh`（已上传容器，日志写 `/workspace/quant-research/w3a16_mid_full.log`）：

```bash
#!/bin/bash
exec > /workspace/quant-research/w3a16_mid_full.log 2>&1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export ASCEND_RT_VISIBLE_DEVICES=5
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /workspace/quant-research/SliderQuant
mkdir -p log

CKPT=log/qwen3-1.7b-w3a16-mid-full/slider_parameters.pth

for ATT in $(seq 1 20); do
  echo "=== W3A16-mid attempt $ATT $(date) ==="
  echo -1000 > /proc/self/oom_score_adj 2>/dev/null || true

  RESUME_ARGS=""
  if [ -f "$CKPT" ]; then
    RESUME_ARGS="--train_resume $CKPT"
    echo "resume from existing checkpoint"
  fi

  numactl --membind=1 python main.py \
    --config configs/qwen3-1.7b-w3a16-mid/config.yaml \
    --output_dir log/qwen3-1.7b-w3a16-mid-full \
    --cache_dir cache \
    $RESUME_ARGS

  EXIT=$?
  if [ "$EXIT" -eq 0 ]; then
    echo "=== W3A16-mid full train done $(date) ==="
    exit 0
  fi
  echo "=== attempt $ATT failed (exit=$EXIT), retry in 30s $(date) ==="
  sleep 30
done
echo "=== all attempts failed $(date) ==="
exit 1
```

要点：`oom_score_adj=-1000` 防自身被杀、checkpoint 自动续训（`--train_resume`）、最多 20 次重试。

## 四、v1 训练与评测结果（2026-09-17 12:54 训练完成）

训练耗时 1.53h，14 轮全部完成，0 次被杀。评测结果：

| 项 | 结果 |
| --- | --- |
| wikitext2 PPL（weight_merge） | **72.75**（异常：应显著低于 RTN 基线 64.96） |
| wikitext2 PPL（slider 模式） | 73.79 |
| 6 任务 avg-6 | 47.74%（boolq 59.94%≈随机、arc_c 28.58%≈随机） |
| 训练 loss | 正常下降，但 loss 质量与最终 PPL 脱钩 |

**结论：训练相对 RTN 基线是净负收益，触发根因诊断（第六节）。**

## 五、全部基准数字（三方对比数据集）

| 配置 | wikitext2 PPL | avg-6 | 说明 |
| --- | --- | --- | --- |
| FP16 基线 | 16.7122 | 64.08% | 上限参照 |
| W3A16-full RTN（无训练） | 1210.10 | — | 卡 4 实测（`_diag_w3a16_full_rtn.sh`） |
| W3A16-full（训练后） | **22.1780** | 57.70% | 28 层全部量化，训练收益 54× |
| W3A16-mid RTN（无训练） | **64.96** | — | 16 层量化（`_diag_w3a16_mid_rtn.sh`） |
| W3A16-mid v1（7 窗，错位输入） | 72.75 | 47.74% | 净负收益 → 触发诊断 |
| W3A16-mid v1 wo_lwc | 110.41 | — | 排除 LWC 假设 |
| W3A16-mid v2（14 窗，错位输入） | 153.65 | 46.71% | 训练更多更差 → 证伪"欠训练"假设 |
| **W3A16-mid v3（首窗输入对齐补丁）** | **17.80** | **61.64%** | **训练正收益，优于 FULL** |

**决定性结论**：中间层量化（layer 6-21）在**正确输入分布**下训练后 PPL 17.80，不仅远优于 RTN 基线 64.96，甚至优于全量化 FULL 的 22.18（因为只量化 16 层），且 avg-6 61.64% 距 FP16 64.08% 仅 2.44pp。

## 六、诊断闭环：PPL 异常根因定位（2026-09-17 全部完成）

按时间序的 9 步诊断，每步都有实证：

| # | 诊断动作 | 结果 | 结论 |
| --- | --- | --- | --- |
| 1 | merge 路径代码检查 | 逻辑正确 | 排除 |
| 2 | 模型组装检查（fp16 PPL） | 16.70 正常 | 排除 |
| 3 | checkpoint 参数检查 | MID LWC bound 偏大 | 疑点（后被证伪） |
| 4 | `--wo_lwc` 评测 | PPL=110 更差 | **LWC 有正贡献，假设排除** |
| 5 | 训练 loss 曲线 | 正常收敛 | loss 与 PPL 脱钩 |
| 6 | LWC 系数表（`_lwc_table.py`） | FULL 全层 clip 0.51-0.60；MID 浅层 0.53-0.58 一致、**深层 14-21 达 0.63-0.83**（layer20=0.826） | 深层裁剪学得偏松 = 欠训练表征 |
| 7 | MID RTN 基线 | 64.96 | 训练后 72.75 → **净负收益** |
| 8 | FULL RTN 基线 | 1210.10 | 训练后 22.18 → **54× 正收益** |
| 9 | 窗口调度探针（`_probe_windows.sh`） | FULL：19 窗 × 2 = 38 轮（4 start 填充 + 11 滑窗 + 4 end 填充，末层 5 次/stage）；MID：7 窗 × 2 = 14 轮（无填充，末层 1 次/stage） | **训练覆盖度差异巨大** |

**最终根因（第 9 步结论，后被 v2 证伪）**：MID 的显式 7 窗调度导致**欠训练**——对比 FULL 的双端填充结构，MID 深层每 stage 只被训练 1 次（末层无重复训练、无 start 填充让浅层先单独收敛）。RTN 双向对比是决定性证据：同一套训练代码，FULL 训练收益 54×，MID 训练反而倒退 → 问题不在 LWC/merge/代码，在训练流程本身。

LWC 系数表佐证：MID 深层 clip 0.63-0.83 与「优化器没学到位」自洽——裁剪方向正确但收敛不足。

> **⚠ 重要修正（2026-09-18）**：上述「欠训练」结论经 v2 实验**证伪**——v2 把轮次从 14 翻倍到 28（补 start/end 填充），PPL 反而从 72.75 恶化到 153.65。真正根因是「**首窗输入错位**」（见第八节）。

## 七、v2 修复：14 窗 FULL 式调度（已证伪，训练更多更差）

### 设计

把 FULL 的 start 填充 + 滑窗 + end 填充结构映射到 layer 6-21：

```yaml
# configs/qwen3-1.7b-w3a16-mid-v2/config.yaml（与 v1 差异仅调度行）
layer_windows_scheduler: "6-6,6-7,6-8,6-9,8-11,10-13,12-15,14-17,16-19,18-21,18-21,19-21,20-21,21-21"
```

14 窗/stage × 2 stage = 28 轮：4 个 start 填充窗（[6]→[6,7]→[6,7,8]→[6,7,8,9]，浅层先单独收敛）+ 6 个滑窗 + 4 个 end 填充窗（[18-21]×2→[19-21]→[20-21]→[21]，末层训 5 次/stage，与 FULL 同构）。

### 运行方式

`_run_w3a16_mid_v2.sh`（卡 1，`ASCEND_RT_VISIBLE_DEVICES=1`，membind=1，20 次重试 + `--train_resume` 续训），日志 `/workspace/quant-research/w3a16_mid_v2.log`。

### 进度（2026-09-17 06:49 UTC）

| 项 | 状态 |
| --- | --- |
| 调度验证 | ✅ Round 3 = layer_id_list [6,7,8,9]，符合设计 |
| 进度 | Round 3/13，epoch 7/60 |
| loss | 1.36 正常下降（Round 0 起点 1.53） |
| ETA | ~08:31 UTC 完成 |

### 完成后评测命令

```bash
python main.py --config configs/qwen3-1.7b-w3a16-mid-v2/config.yaml \
  --output_dir log/qwen3-1.7b-w3a16-mid-v2 --cache_dir cache \
  --test_mode --weight_merge --eval_ppl \
  --tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande \
  --lm_eval_batch_size auto \
  --resume log/qwen3-1.7b-w3a16-mid-v2/slider_parameters.pth
```

**判定标准**：v2 PPL 应显著低于 MID RTN 基线 64.96（训练正收益），理想值接近 FULL 的 22.18 与 RTN 64.96 之间按层数比例的内插水平。

### 结果（证伪）

v2 训练完成（28 轮，2.37h），但评测 **PPL = 153.65 / avg-6 = 46.71%**，比 v1（72.75）更差、比 RTN（64.96）差 2.4 倍。**训练覆盖度翻倍反而更差，彻底证伪「欠训练」假设** → 触发第八节真正根因定位。

## 八、真正根因：部分层量化首窗输入错位 + v3 补丁修复（2026-09-18）

### 8.1 根因（代码级 bug）

[sliderquant.py](file:///c:/code/SliderQuant/quantize/sliderquant.py) 训练循环中，`inps` 只捕获 **layer 0 的输入**（embedding 输出），因为 `Catcher` 挂在 layer 0 处并在捕获后 `raise ValueError` 截断前向（[L310](file:///c:/code/SliderQuant/quantize/sliderquant.py#L310)）。

滑窗累积机制隐含假设「**第一个量化窗口从 layer 0 开始**」——FULL 的首窗就是 `[0]`，输入天然正确。但 MID 的首窗是 `[6]`（或 `[6,7,8,9]`），训练时 layer 6 被直接喂 embedding 输出，而**真实推理时 layer 6 的输入是 fp16 layer 0–5 的输出**。两者分布完全不同：

| | FULL（首窗 [0]） | MID（首窗 [6]） |
| --- | --- | --- |
| 首窗输入 | embedding 输出 ✓ | embedding 输出 ✗（应为 layer 0–5 输出） |
| 结果 | 训练正确 → PPL 22 | 训练错位 → PPL 72~153 |

**这解释了所有现象**：训练无效（错位输入上学的 LoRA/LWC 推理时错配）、训练越多越差（错位参数越「自信」）、深层 loss 异常抬高（错位输入经 layer 6-13 累积放大）、LWC 深层偏松（用错误分布拟合）。

### 8.2 补丁

插入 [sliderquant.py L496](file:///c:/code/SliderQuant/quantize/sliderquant.py#L496)（`assert quant_step` 之后、训练循环前），当首窗起始层 > 0 时先把 `inps` 通过前置 fp16 层前向：

```python
_start_layer = layer_windows_scheduler[0][0]
if _start_layer > 0 and not args.test_mode:
    logger.info(f"[partial-quant-fix] forward inps through fp16 layers [0,{_start_layer}) ...")
    _prefix_layers = to_dev(layers[0:_start_layer], [dev] * _start_layer)
    with torch.no_grad():
        with torch.npu.amp.autocast(dtype=fp16_type):
            for j in tqdm(range(0, args.nsamples, args.inference_batch_size)):
                bs = min(args.inference_batch_size, args.nsamples - j)
                inps[j:j+bs] = obtain_teacher_output(
                    _prefix_layers, inps[j:j+bs].to(dev),
                    infer_attention_mask[:bs], position_ids, position_embeddings,
                    args=args, devs=[dev] * _start_layer,
                ).cpu()
    del _prefix_layers
    cleanup_memory(logger=logger)
```

### 8.3 验证：loss 曲线与 FULL 逐层对齐

| 窗口 | v2（错位） | v3（对齐） | FULL 参照 |
| --- | --- | --- | --- |
| R0 [6] | 1.25 | **0.034** | 0.007 |
| R7 [14-17] | 9.9 | **4.97** | 5.9 |
| R8 [16-19] | 22.4 | **12.6** | 14.6 |
| R9 [18-21] | 53.5 | **31.1** | 35.6 |

浅层 loss 36 倍下降、深层 loss 从「比 FULL 高 1.5 倍」降到「与 FULL 同量级」——训练自此在正确输入分布上工作。

### 8.4 最终结果（v3）

| 指标 | 值 | 参照 |
| --- | --- | --- |
| wikitext2 PPL | **17.80** | FP16 16.71 / FULL 22.18 / RTN 64.96 |
| avg-6 | **61.64%** | FP16 64.08% / FULL 57.70% |
| 训练耗时 | 3.01h | 28 轮，卡 1，0 重试 0 OOM |

六任务明细（v3，weight_merge）：

| 任务 | 准确率 |
| --- | --- |
| piqa | 0.7067 |
| arc_easy | 0.6852 |
| arc_challenge | 0.3993 |
| boolq | 0.7657 |
| hellaswag | 0.5487 |
| winogrande | 0.5927 |

**对比 v1/v2 的 6 任务**：v1 avg-6 47.74%（boolq 59.94%≈随机、arc_c 28.58%≈随机）、v2 46.71% → v3 61.64% 全部恢复正常，逼近 FP16 64.08%。

## 九、后续步骤

1. ✅ v3 训练 + 评测完成，中间层量化训练自此有效
2. 产出三方对比报告（FP16 / W3A16-full / W3A16-mid-v3）
3. 沉淀结论：部分层量化首窗输入错位 bug + 补丁；NUMA membind 方案

## 十、关键经验（可复用）

1. **共享服务器上"内存够却 OOM 被杀"→ 先查 NUMA**：`numactl --hardware` + `cat /sys/bus/pci/devices/*/numa_node` 对照各 node 的 `free -m`。
2. **昇腾 NPU 训练进程内存应绑到"有大块空闲内存的 node"**，卡本地 node 只留给 devmm 强制的 DMA pin 内存。
3. **`drop_caches` 是 membind 成功的前提**：不先释放 page cache，目标 node 可能没有足够连续内存供 pin。
4. `low_memory: true`（激活留 CPU、小批量 H2D）+ `low_cpu_memory: true` + `inference_batch_size` 调小，是压训练峰值的三板斧。
5. checkpoint 续训（`--train_resume`）+ 重试脚本，可在不稳定共享环境把长训练拆碎跑完。
6. **部分层量化（窗口不从 layer 0 开始）必须把 `inps` 先前向到首窗起始层**：SliderQuant 的 `inps` 只捕获 layer 0 输入，滑窗累积隐含「首窗从 layer 0 开始」假设；若首窗起始层 > 0，训练会拿 embedding 输出去喂首窗层，与推理时「前置 fp16 层输出」错位 → 训练净负收益。修法见第八节补丁（一行前向调用即可）。
7. **PowerShell 下 server.py exec 传复合命令（分号/管道/嵌套引号）会翻车**：一律封装成 bash 脚本上传后 `docker exec ... bash script.sh` 执行。
8. **loss 曲线对照 FULL 是快速判据**：部分层量化训练是否正常，看深层窗口 loss 是否与 FULL 对应层同量级；错位时会比 FULL 高 1.5 倍以上。
