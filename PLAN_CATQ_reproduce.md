# CAT-Q 复现 — 进展与执行计划

> 目标：复现 CAT-Q（三值量化 PTQ，ICML 2026 Oral）论文结果，模型 Qwen3-1.7B，配置 W1.58A16。
> 状态：**路径 A（官方 checkpoint 评测）已完成；路径 B（自实现训练）待继续**。

---

## 一、任务目标

复现论文 *CAT-Q: Cost-efficient and Accurate Ternary Quantization for LLMs*（arXiv:2606.26650，ICML 2026 Oral，Intel Labs China）。

- 把 Qwen3-1.7B 权重三值化到 `{-1, 0, +1}`（W1.58A16），用 PTQ（512 校准样本，非 QAT）
- 复现路径分两步（用户已选 **C + Qwen3-1.7B + W1.58A16**）：
  - **路径 A（已完成）**：官方 checkpoint 评测，拿到复现基准
  - **路径 B（待做）**：自实现 LM+ST 训练，与官方基准对齐

**复现基准（官方 checkpoint 评测，avg-5）：**

| 任务 | 准确率 |
|---|---|
| piqa | 0.6795 |
| arc_easy | 0.5497 |
| arc_challenge | 0.3114 |
| hellaswag | 0.4589 |
| winogrande | 0.5612 |
| **avg-5** | **0.5121** |

（Qwen3-1.7B FP16 的 avg-5 = 61.38%，三值化后 51.21%，退化 ~10pp）

---

## 二、CAT-Q 方法（完整公式，已全部提取）

### LM（Learnable Modulation，3 个可学习因子）

设分组（group_size=128）内：

- 均值 μ0 = mean(W)，尺度 α0 = absmean(W − μ)，阈值 Δ0 = init_round_thd = 0.5

三个可学习因子调制三者：

```
μ = μ0 + δμ · α0        δμ = sigmoid(mu_bound)·2 − 1  ∈ [−1, 1]
α = δα · α0            δα = sigmoid(scale_bound)·2    ∈ [0, 2]
Δ = δΔ · Δ0            δΔ = sigmoid(round_bound)·2
```

### ST（Softened Ternarization，软化三值化）

训练侧用可微 tanh 过渡函数（软三值化），随训练进度 s 渐进增大 → 逼近硬三值化：

```
Ŵ = W − μ
T = f(Ŵ; s, Δ) = [ tanh(s(Ŵ−Δ)) + tanh(s(Ŵ+Δ)) ] / [ 2·tanh(s) ]
```

- s（sharpness）初始 s0=30，按 progressive_ratio=0.8、phi_x_n=3 渐进 schedule 增大
- 推理侧退化为硬三值化：`T = clamp(round((W−μ)/α · 0.5/Δ), −1, 1)`
- 前向反量化：`W_q = α·T + μ`（训练侧 drop_quant_mu=true 时不加 μ，推理时加回 μ）

### 其他

- 复用 SliderQuant 的 sliding-layer 重构（fill_window_size=4, num_layer=4, sliding_layer=2, quant_step=1）
- LoRA rank r=64（远大于普通 2/3/4-bit 的 r=4，三值信息损失大需强补偿）
- huber loss、grad_clip=1.0

---

## 三、已完成进展（截至本会话）

### 1. 代码仓库分析
- clone 了 `https://github.com/IntelChina-AI/BitTern` 到 `c:\code\BitTern`
- 核心目录 `projects/cat-q/`，含 `quantize/quantizer.py`（TernaryQuantizer 推理侧完整实现）、`merge.py`、`checkpoint.py`、`main.py`
- 关键结论：**官方只开源了 checkpoint + 推理 + 评测 + GGUF 部署，训练代码未开源**（README 明确 "Training code will be released separately"）

### 2. 官方 checkpoint 下载
- 服务器 `/workspace/quant-research/catq_checkpoints/qwen3-1.7b/`：
  - `parameters.pth`（411 MB，CAT-Q 可学习参数）
  - `config.yaml`（官方量化配置）
  - `Qwen3-1.7B-catq-q2_0.gguf`（1 GB，打包三值权重）

### 3. NPU 适配 + 官方评测跑通（路径 A）
- 本地修改 `c:\code\BitTern\projects\cat-q\`：
  - `models/LMClass.py`：`torch.device("cuda"...)` → `npu`，加 `import torch_npu`
  - `quantize/utils.py`：HFLM 加 `device=str(lm.device)`；去掉 piqa 的 `load_yaml_config` hack（lm_eval 新版移除该 API）
- 打包上传到服务器 `/workspace/quant-research/cat-q/`
- 评测命令（卡 0，避开被占用的卡 3/5）：
  ```bash
  python main.py --config configs/qwen3-1.7b/config.yaml \
    --model /workspace/quant-research/model_zoo/Qwen3-1.7B \
    --checkpoint configs/qwen3-1.7b/parameters.pth \
    --tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande \
    --lm_eval_batch_size auto:4
  ```
- 结果：**avg-5 = 0.5121**（即上面基准表）

---

## 四、剩余计划（待新对话继续）

### c5：自实现 LM+ST 三值化训练（路径 B）

在现有 **SliderQuant** 框架（`c:\code\SliderQuant`，已 NPU 适配）上新增 `cat-q` 量化模式。官方 cat-q 的 `TernaryQuantizer`（推理侧）可直接参考复制，只需补上 **ST 训练侧 tanh 软化 + s schedule**。

实现步骤：

1. **`quantize/quantizer.py` 新增 `TernaryQuantizer`**（参考 `c:\code\BitTern\projects\cat-q\quantize\quantizer.py` L68-118）：
   - LM 3 因子：`generate_mu_factor/scale_factor/round_factor`（sigmoid 参数化，`requires_grad=True`）
   - ST 训练侧：`T = [tanh(s(Ŵ−Δ)) + tanh(s(Ŵ+Δ))] / 2tanh(s)`，s 随训练 step 从 s0 渐进增大
   - 推理侧：硬三值化 `clamp(round(Ŵ/α·0.5/Δ), −1, 1)`
2. **`quantize/int_linear.py` 支持 wbits=1**：wbits=1 时用 `TernaryQuantizer`（否则用原 `UniformAffineQuantizer`）
3. **`models/int_llama_layer.py` 支持 `quant_mode=="cat-q"`**：weight_merge 时用 TernaryQuantizer 的硬三值化
4. **`main.py` 新增参数**：`learnable_scale/mu/round`、`learnable_factor_act`、`init_round_thd`、`progressive_ratio`、`phi_x_n`、`s0`、`shift_mu`、`drop_quant_mu`、`ter_scale_type`
5. **新建 config** `configs/qwen3-1.7b-catq/config.yaml`（关键参数见下）
6. **训练** → **weight_merge 评测**（PPL + 5 任务），对比目标 avg-5 ≈ 0.5121

### c6：评测对比

- 自实现训练结果 vs 官方 checkpoint 结果（avg-5 0.5121）
- 目标：±0.2pp 内对齐（README 说明代码 refactor 后 checkpoint 精度可能 ±0.2pp）

---

## 五、官方 config（关键参数，供 c5 直接使用）

`c:\code\BitTern\projects\cat-q\configs\qwen3-1.7b\config.yaml` 关键内容：

```yaml
model: Qwen/Qwen3-1.7B
wbits: 1            # 三值
abits: 16
group_size: 128
quant_mode: cat-q
use_scaling: false

epochs: 60
nsamples: 512
batch_size: 9
inference_batch_size: 12
calib_dataset: c4
use_bfloat16: true
low_memory: true

grad_clip: 1.0
loss_function: huber
fill_window_size: 4
init_round_thd: 0.5
progressive_ratio: 0.8
phi_x_n: 3
s0: 30.0
learnable_factor_act: sigmoid
learnable_scale: true
learnable_mu: true
learnable_round: true
shift_mu: true
drop_quant_mu: true

num_layer: 4
sliding_layer: 2
quant_step: 1
last_round_inp_num: 1

r: 64
lora_quant: true
lora_lr: 0.0003
learnable_factor_lr: 0.0015
warmup_ratio: 0.0
```

---

## 六、关键文件与路径

### 本地
| 内容 | 路径 |
|---|---|
| SliderQuant 框架（已 NPU 适配） | `c:\code\SliderQuant` |
| BitTern 仓库（cat-q 官方代码） | `c:\code\BitTern\projects\cat-q` |
| NPU server 脚本 | `c:\code\SliderQuant\.trae\skills\npu-server\server.py` |
| 本计划 | `c:\code\SliderQuant\PLAN_CATQ_reproduce.md` |

### 服务器（121.37.53.41:10001，容器 sliderquant-qwen3）
| 内容 | 路径 |
|---|---|
| SliderQuant（已 NPU 适配 + 首窗补丁） | `/workspace/quant-research/SliderQuant` |
| cat-q 代码（已 NPU 适配） | `/workspace/quant-research/cat-q` |
| 官方 checkpoint | `/workspace/quant-research/catq_checkpoints/qwen3-1.7b/` |
| Qwen3-1.7B 模型 | `/workspace/quant-research/model_zoo/Qwen3-1.7B` |
| Qwen3-8B 模型 | `/workspace/quant-research/model_zoo/Qwen3-8B` |
| 评测日志 | `/workspace/quant-research/catq_eval.log` |

---

## 七、环境要点（重要）

1. **服务器连接**：`uv run --with paramiko python "c:\code\SliderQuant\.trae\skills\npu-server\server.py" {check,exec,upload,download}`
2. **卡占用**：8× Ascend 910B3（64GB HBM/卡），7 个容器共享。**当前卡 0 空闲**（卡 3/5 被其他容器占用）。评测/训练用 `ASCEND_RT_VISIBLE_DEVICES=0`
3. **内存**：宿主 node 1 需 `drop_caches` + `numactl --membind=1`
4. **HF 镜像**：`export HF_ENDPOINT=https://hf-mirror.com`、`HF_HUB_DISABLE_XET=1`
5. **lm_eval 版本**：新版无 `load_yaml_config`（cat-q 评测代码已适配）
6. **PowerShell 引号**：server.py exec 复合命令会翻车，一律封装 bash 脚本上传执行
7. **Write/SearchReplace 工具会偶发超时但实际已写入**，写后 Read 确认即可

---

## 八、已知结论（复现前参考）

- CAT-Q 三值化：Qwen3-1.7B avg-5 = 51.21%（FP16 61.38%，退化 ~10pp）
- 训练 tokens 比 BitNet 1.58-bit 少 ~100,000×（512 samples vs 100B tokens）
- 三值权重：2bit 存储 + 每 128 组一个 fp16 scale（GGUF q2_0，Bonsai runtime）
- 与 SliderQuant 同团队（Intel Labs China, Anbang Yao），CAT-Q 就是基于 SliderQuant 实现
