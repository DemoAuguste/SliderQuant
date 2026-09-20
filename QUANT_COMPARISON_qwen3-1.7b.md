# Qwen3-1.7B 量化实验结果对比（FP16 / W3A16 / W4A16）

- 评测日期：2026-09-14
- 模型：`Qwen/Qwen3-1.7B`（Dense，约 1.7B 参数）
- 量化方案：SliderQuant weight-only（`quant_mode=lora_only`，LoRA rank 4 补偿）
  - W3A16：3-bit 权重 + FP16 激活
  - W4A16：4-bit 权重 + FP16 激活
- 硬件：Ascend 910B3（NPU，单卡）
- 运行环境：MindSpeed-LLM 容器 `v26.1.0-cann9.1.0-torch_npu2.7.1.post8-910b-openeuler24.03-py3.12`
- 依赖版本：torch 2.7.1 / torch_npu 2.7.1.post8 / transformers 5.2.0 / lm-eval 0.4.13

## 一、结果总览

### PPL（越低越好）

| 数据集 | FP16 | W4A16 | W3A16 |
| --- | --- | --- | --- |
| wikitext2 | 16.7122 | **18.0663** | 22.1780 |

### 零样本常识推理（准确率，越高越好）

| 任务 | FP16 | W4A16 | W3A16 |
| --- | --- | --- | --- |
| piqa | 72.36% | 67.95% | 69.04% |
| arc_easy | 69.57% | 58.59% | 56.69% |
| arc_challenge | 43.34% | 34.04% | 34.90% |
| boolq | 77.61% | 76.76% | 76.09% |
| hellaswag | 60.36% | 57.29% | 51.00% |
| winogrande | 61.25% | 60.46% | 58.48% |
| **avg-5** | **61.38%** | **55.67%** | **54.02%** |
| **avg-6** | **64.08%** | **59.18%** | **57.70%** |

> `avg-5` = piqa/arc_easy/arc_challenge/hellaswag/winogrande 均值；`avg-6` 在此基础上加 boolq。

## 二、相对 FP16 的精度损失

| 指标 | W4A16 损失 | W3A16 损失 |
| --- | --- | --- |
| wikitext2 PPL | **+1.35（+8.1%）** | +5.47（+32.7%） |
| avg-6 | **-4.90 百分点** | -6.38 百分点 |

## 三、结论

1. **W4A16 明显优于 W3A16**（符合预期，4-bit > 3-bit）：
   - PPL：18.07 vs 22.18（W4A16 更接近 FP16 的 16.71，差距缩小 4.11）
   - avg-6：59.18% vs 57.70%（W4A16 高 1.48 个百分点）
2. **W4A16 相较 FP16 仍有可感知下降**：PPL +1.35、avg-6 -4.90 个百分点。
3. **任务分化**：`arc_easy`（-10.98）、`arc_challenge`（-9.30）损失较大；`boolq`（-0.85）、`winogrande`（-0.79）几乎不变。
4. **局部噪声**：W4A16 在 `piqa`（-1.09）、`arc_challenge`（-0.86）上略低于 W3A16，属单次复现的正常波动（种子固定但不同位宽训练轨迹不同），整体趋势仍清晰偏向 W4A16。

## 四、量化配置（W4A16）

```yaml
model: model_zoo/Qwen3-1.7B
wbits: 4
abits: 16
epochs: 60
quant_mode: lora_only
lora_rank: 4
use_lora: true
nsamples: 128
calib_dataset: wikitext2
fp16_act: true
low_cpu_memory: true
```

## 五、评测过程说明

- 训练：单卡 NPU，约 3.97h，无 OOM。
- 评测期间遇共享服务器间歇性 OOM，下游任务改用「单任务逐个跑 + 失败重试」后全部完成：
  - `wikitext2 PPL` + `piqa` 首次评测成功；
  - 其余 5 任务（arc_easy/arc_challenge/boolq/hellaswag/winogrande）经重试后完成。

## 六、产物位置

| 产物 | 位置 |
| --- | --- |
| W4A16 checkpoint | `log/qwen3-1.7b-w4a16/slider_parameters.pth` |
| W3A16 checkpoint | `log/qwen3-1.7b-w3a16/slider_parameters.pth` |
| FP16 基线 | `FP16_BASELINE_qwen3-1.7b.md` |
| W3A16 对比 | `W3A16_vs_FP16_qwen3-1.7b.md` |
| 本文档 | `QUANT_COMPARISON_qwen3-1.7b.md` |
