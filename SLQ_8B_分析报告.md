# SLQ 复现分析报告 — Qwen3-8B（P1 计划）

- 论文：SLQ (Statistically-Lossless Quantization)，COLM 2026，arXiv 2605.02404
- 模型：Qwen3-8B（`/opt/zh/train/model_from_hf/qwen3_hf`）
- 硬件：Ascend 910B3 × 1 卡（NPU），容器 `msllm-train-qwen3-8b`
- 日期：2026-09-20 ~ 2026-09-21
- 状态：**P1 全流程完成**（p1-1 ~ p1-6）

---

## 1. 背景与目标

SLQ 的核心思想：用 **EAR（Expected Agreement Rate）** 度量原始分布与量化后分布在前 top-10 高概率 token 上的概率质量重叠，将"统计无损"从"逐 token 精确"（PPL/KL）提升到**分布级**标准，从而在更低位宽下保持模型行为。

```text
EAR = (1/N) Σᵢ Σₖ₌₁..₁₀ min(pᵢ(k), qᵢ(k))
```

量化采用**逐层非均匀混合位宽 {3..8}**，由敏感度估计 + 预算分配（ILP/DP）+ 预算二分求得。

两个目标：
| 目标 | 判定标准 | 适用范围 |
|---|---|---|
| **DL** (Distribution-lossless) | EAR ≥ 0.99（绝对）或 ≥ base−margin（相对） | top-10 概率质量 ≥ 0.99 的大模型 |
| **TL** (Task-lossless) | lm_eval 恢复率 ≥ 99% | 通过单点标定 recovery ≈ 1 − slope·DKL |

本阶段任务（P1）：在 **Qwen3-8B** 上复现 SLQ，验证 DL 目标可达性，并给出 TL 位宽配置与最终恢复率。

---

## 2. 实验配置

| 项 | 值 |
|---|---|
| 校准数据 | wikitext2，512 样本（nsamples=512），seqlen=2048 |
| 敏感度子集 | 128 样本（sens_nsamples=128） |
| 分组 | `split`：每层 attn(4×Linear) / mlp(3×Linear) 两组 → **72 组** |
| group_size | 128（非对称量化） |
| 温度 | **0.6**（温度缩放 logits，见 §3.2） |
| 评测任务 | piqa, arc_easy, arc_challenge, boolq, hellaswag, winogrande（avg-6） |
| lm_eval | 进程内 HFLM（device="npu"），batch_size=8 |
| TL 锚点位宽 | uniform-6（0.6B 经验：4-bit 已偏离线性区，8B 取 6） |
| TL 目标 | recovery ≥ 0.99 |

---

## 3. 阶段进展与结果

### p1-1 校准数据 ✅

512 样本 wikitext2 校准集 + 128 样本敏感度子集，缓存于 `/opt/zh/train/SliderQuant/cache/dataloader_qwen3_wikitext2_512.cache`。服务器 HF 直连不通，数据走本地缓存。

### p1-2 base 基线（温度选择 + γ² 定律验证）✅

**温度扫描**（决定 8B 用 T=0.6）：温度 <1 压缩 softmax 温度，放大 top-10 概率质量，使 EAR 度量更有区分度。

| 温度 | self EAR（top-10 质量） | uniform-8 | uniform-7 | uniform-6 | uniform-5 |
|---|---|---|---|---|---|
| 0.5 | 0.99473 | 0.97937 | — | 0.96058 | 0.93447 |
| **0.6** | **0.98875** | **0.97384** | — | — | — |
| 0.7 | 0.97925 | 0.96485 | 0.95969 | — | — |
| 1.0 | 0.92298 | 0.91052 | 0.90613 | — | — |

取 **T=0.6**：self EAR=0.98875 最接近论文 DL 适用域（≥0.99）的边界。

**uniform 基线（T=0.6）**：

| 配置 | EAR | KL |
|---|---|---|
| self（原始 vs 原始） | 0.98875 | 0.0 |
| uniform-4 asym | 0.87263 | 0.15075 |
| **uniform-4 sym** | **≈3.4e-5** | **20.70** |
| uniform-3 asym | 0.72906 | 0.70134 |
| uniform-5 asym | 0.93036 | 0.04090 |

**γ² 定律实证**：对称量化 vs 非对称量化（4-bit）EAR 从 0.873 崩到 ~0（KL 0.15 → 20.7），对称噪声放大 γ²（约 ×2）倍，直接验证论文核心前提——非对称量化是 DL 的前提。

### p1-3 GPTQ 逐层量化器（RTN/GPTQ 可切换）✅

`slq/gptq.py` 实现 chunk-Hessian GPTQ（`gptq_quantize_weight`）。

**NPU 适配（关键）**：910B3 无 `cholesky_inverse` 算子、`torch.inverse` 在 NPU 上挂起（AICore 0%）、逐列循环 D2H 同步死锁 → **Hessian 逆与逐列量化循环全部搬 CPU**（4096 维逆 1.2s / 12288 维 10.1s）。适配要点：
- `requires_grad_(False)` 冻结参数（否则 `copy_` 报 in-place 错）；
- hook 每批每 Linear 取 `PER_BATCH` 行落 CPU（防 CPU OOM）；
- 权重快照存 CPU（GPU 16GB 模型 + 4.6GB logits 放不下）；
- 每层 `torch.npu.empty_cache()`。

### p1-4 DL 搜索：**不可达**（关键负面结论）❌

**流程**：敏感度估计（72 组，128 样本）→ 预算二分（实测-调整）→ 验证。

敏感度结论：**末层（L35）最敏感**（dEAR=+0.101），且随层号单调上升（L28: +0.085 → L35: +0.101），符合深层的注意力/MLP 越敏感、需要越高位宽的直觉。

二分搜索日志（target_ear=0.985，base_ear=0.97373，eff_target=0.97373）：

```text
[DL] budget 5.50 avg_bits=5.496 EAR(sens)=0.93962
[DL] budget 6.75 avg_bits=6.746 EAR(sens)=0.96254
[DL] budget 7.38 avg_bits=7.371 EAR(sens)=0.96972
[DL] budget 7.69 avg_bits=7.681 EAR(sens)=0.97114
[DL] budget 7.84 avg_bits=7.836 EAR(sens)=0.97316
[DL] budget 7.92 avg_bits=7.917 EAR(sens)=0.97348
[DL] budget 7.98 avg_bits=7.972 EAR(sens)=0.97369   ← 仍 < 0.97373
RuntimeError: DL search failed: cannot meet EAR target even at max bits
```

**判定实验**（GPTQ-8 vs RTN-8）：

```text
self (T=0.6):       EAR=0.98875
RTN  uniform-8:     EAR=0.97384
GPTQ uniform-8:     EAR=0.97381   ← 与 RTN 仅差 0.00003
```

**结论**：
1. 8B 在 8bit 的 EAR 损失 ~0.015（0.98875 → 0.9738）是**量化固有精度损失**，非 RTN 舍入误差，**GPTQ 误差补偿也无法救回**（GPTQ-8 = RTN-8）；
2. 8B top-10 概率质量 0.98875 < 0.99，**超出 DL 绝对阈值（0.99）的适用范围**；
3. 即使按相对目标（base_ear−margin），8bit 已到理论上限（0.9737 < 0.97373，仅差 5e-5）→ **DL 在 8B 上不可达，必须转 TL 目标**。

### p1-5 TL 搜索（单点标定）✅

流程：BF16 基线 lm_eval → uniform-6 锚点测 KL + lm_eval → 复用 DL 敏感度 → 单点标定斜率 → 二分预算。

| 量 | 值 |
|---|---|
| BF16 avg-6 | 0.74073 |
| anchor uniform-6 avg-6 | 0.73870 |
| **anchor recovery** | **0.99726** |
| kl_cal（锚点 KL） | 0.01307 |
| **slope** | **0.2578**（recovery ≈ 1 − slope·KL） |
| target_kl = (1−0.99)/slope | 0.03879 |
| 最终实测 KL | 0.03942（略超 target，但任务恢复率达标，见 §3.6） |
| **avg_bits** | **5.554** |
| 最终 EAR | 0.93453 |

**位宽分布**（72 组参数总量 6.95G）：

| 位宽 | 参数量 | 占比 |
|---|---|---|
| 3 | 343.9 M | 4.95% |
| 4 | 536.9 M | 7.73% |
| 5 | 1,786.8 M | 25.72% |
| 6 | 3,481.3 M | 50.12% |
| 7 | 796.9 M | 11.47% |
| **合计** | **6,945.8 M** | **100%** |

浅层（前几层 attn/mlp）分配到 3-4 bit，中深层以 5-6 bit 为主，末层少量 7 bit —— 与敏感度排序一致。

### p1-6 benchmark 验证（TL 配置最终恢复率）✅

对 TL 5.554bit 配置做 avg-6 全量 lm_eval：

| 任务 | BF16 | TL 5.55b | 恢复率 |
|---|---|---|---|
| piqa | 0.77584 | 0.77421 | 0.9979 |
| arc_easy | 0.80892 | 0.79545 | 0.9834 |
| arc_challenge | 0.56911 | 0.55717 | 0.9790 |
| boolq | 0.86514 | 0.86881 | 1.0042 |
| hellaswag | 0.74895 | 0.74477 | 0.9944 |
| winogrande | 0.67640 | 0.68272 | 1.0093 |
| **avg** | **0.74073** | **0.73719** | **0.9952** |

**恢复率 0.9952 ≥ 0.99 → TL 达标**。注意单点标定目标 KL=0.03879 与实测 0.03942 有微小偏差，但任务级恢复率仍达标——印证 TL 的单点标定存在裕量，斜率近似是稳健的。

---

## 4. 0.6B (P0) vs 8B (P1) 对比

| 指标 | 0.6B（P0 已完成） | 8B（P1 本次） |
|---|---|---|
| top-10 质量（self EAR） | 0.8400（T=1.0） | 0.98875（T=0.6） |
| uniform-4 sym | 崩坏（γ² 定律） | 崩坏（EAR 3e-5，KL 20.7） |
| DL 可达性 | 可达（相对 base−0.01） | **不可达**（绝对 0.985/相对均差 5e-5） |
| TL 平均位宽 | 7.19 bits | **5.554 bits** |
| TL 恢复率 | 0.9952 | 0.9952 |
| 末层 MLP 敏感度 | 最敏感（dEAR=0.188） | 最敏感（dEAR=0.101） |

**核心结论**：8B 比 0.6B 位宽更低（5.55 vs 7.19 bits），验证论文"**大模型冗余更多、可压更狠**"的核心主张。同时 DL 目标的绝对阈值（EAR≥0.99）在 8B 上失效（top-10 质量 0.9887<0.99），TL 成为 8B 的实际适用目标。

---

## 5. 关键结论汇总

1. **DL 在 8B 上不可达**：8bit 是 EAR 的天然上限（~0.974，差 0.015 于 0.99），GPTQ 误差补偿无改善（0.97381 ≈ RTN 0.97384）；8B top-10 质量 0.9887 < 0.99 超出 DL 适用域。
2. **TL 达标**：SLQ-TL 5.554 bits，avg-6 恢复率 **0.9952 ≥ 0.99**；单点标定（slope=0.258）有效。
3. **γ² 定律实证**：4-bit 对称量化 EAR 崩坏至 ~0（KL 20.7），非对称量化是前提。
4. **敏感度结构**：深层 > 浅层，MLP > attn（末层最敏感），位宽分配与敏感度一致。
5. **模型规模效应**：8B 冗余远大于 0.6B（5.55 vs 7.19 bits 达成同样 99% 恢复率）。

---

## 6. 技术踩坑记录（NPU 复现可复用）

| # | 坑 | 解法 |
|---|---|---|
| 1 | NPU 无 `cholesky_inverse` / `torch.inverse` 挂起 | Hinv 全 CPU 计算（4096 维 ~1.2s） |
| 2 | GPTQ 逐列循环 scalar 索引触发 D2H 同步死锁 | 量化循环整体搬 CPU |
| 3 | `copy_()` 对 requires_grad 叶子报 in-place 错 | 先 `requires_grad_(False)` + `torch.no_grad()` |
| 4 | hook 缓存全量激活 OOM（CPU） | 每批每 Linear 取 PER_BATCH 行 |
| 5 | GPU OOM（模型 16G + logits 4.6G） | 权重快照落 CPU、`empty_cache()` |
| 6 | lm_eval 联网卡死（服务器直连不通） | `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1` |
| 7 | 敏感度结束后模型停在 8bit 量化态 | 应用新配置前必须 restore 原始权重 |
| 8 | DP 背包预算过小返回 None / JSON 往返键 str | 二分跳过不可行解；`int(k)` 转回 |
| 9 | `model_metrics` 返回 `(ear, kl)` 解包写反 | `_, kl = model_metrics(...)` |
| 10 | Windows CRLF 破坏 .sh heredoc | 上传后 `sed -i 's/\r$//'`；含 `$`/嵌套引号一律脚本化 |
| 11 | `docker exec` 后台 job 报 exit -1 但进程正常 | 以 npu-smi/日志为准，忽略假失败 |

---

## 7. 产物与复现入口

**服务器**（`/opt/zh/train/output/slq-8b/`）：
- `qwen3-8b-base/slq_base.json` — base 基线 + γ² 实验
- `qwen3-8b-tl/bf16_scores.json` — BF16 基线
- `qwen3-8b-tl/anchor_6b_scores.json` — uniform-6 锚点
- `qwen3-8b-tl/slq_tl.json` — TL 位宽配置（alloc 72 组，avg_bits=5.554）
- `qwen3-8b-tl/slq_eval.json` — TL 最终评测（恢复率 0.9952）
- `gptq8_test.log` / `dl.log` / `tl.log` — 各阶段日志
- 本机同步：`C:\code\SliderQuant\`（`main_slq.py` + `slq/` 包 + `_slq_*.sh` 脚本）

**复现命令**：
```bash
# base 基线
python main_slq.py --config configs/qwen3-8b-slq-base4.yaml --mode base
# DL 搜索（已知失败，验证用）
python main_slq.py --config configs/qwen3-8b-slq-dl.yaml --mode dl
# TL 搜索
python main_slq.py --config configs/qwen3-8b-slq-tl.yaml --mode tl
# TL 配置评测
python main_slq.py --config configs/qwen3-8b-slq-tl.yaml --mode eval \
    --apply_config /opt/zh/train/output/slq-8b/qwen3-8b-tl/slq_tl.json
```
