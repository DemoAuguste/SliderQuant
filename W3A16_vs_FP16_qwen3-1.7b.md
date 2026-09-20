# Qwen3-1.7B W3A16 量化结果与基线对比

- 评测日期：2026-09-14
- 模型：`Qwen/Qwen3-1.7B`（Dense，约 1.7B 参数）
- 量化方案：SliderQuant W3A16（3-bit 权重 + FP16 激活，`quant_mode=lora_only`，LoRA rank 4 补偿）
- 硬件：Ascend 910B3（NPU，单卡）
- 运行环境：MindSpeed-LLM 容器 `v26.1.0-cann9.1.0-torch_npu2.7.1.post8-910b-openeuler24.03-py3.12`
- 依赖版本：torch 2.7.1 / torch_npu 2.7.1.post8 / transformers 5.2.0 / lm-eval 0.4.13

## 一、结果对比

### PPL（越低越好）

| 数据集 | FP16 基线 | W3A16 | 变化 |
| --- | --- | --- | --- |
| wikitext2 | 16.7122 | 22.1780 | **+5.4658（+32.71%）** |

### 零样本常识推理（准确率，越高越好）

| 任务 | FP16 基线 | W3A16 | 变化（百分点） |
| --- | --- | --- | --- |
| piqa | 72.36% | 69.04% | -3.32 |
| arc_easy | 69.57% | 56.69% | -12.88 |
| arc_challenge | 43.34% | 34.90% | -8.44 |
| boolq | 77.61% | 76.09% | -1.52 |
| hellaswag | 60.36% | 51.00% | -9.36 |
| winogrande | 61.25% | 58.48% | -2.77 |
| **avg-5** | **61.38%** | **54.02%** | **-7.36** |
| **avg-6** | **64.08%** | **57.70%** | **-6.38** |

> `avg-5` = piqa/arc_easy/arc_challenge/hellaswag/winogrande 均值；`avg-6` 在此基础上加 boolq。

## 二、结论

1. W3A16 在 Qwen3-1.7B 上**成功复现，全链路打通**（训练 → 推理模式 → PPL → 下游任务），但精度有可感知的下降。
2. **PPL 上升 5.47 点**（16.71 → 22.18），**avg-6 下降 6.38 个百分点**。
3. 各任务降幅分化明显：`arc_easy`（-12.88）、`hellaswag`（-9.36）降幅较大；`boolq`（-1.52）、`winogrande`（-2.77）相对稳健。

## 三、分析（为何不"几乎无损"）

1. **模型规模差异**：论文"几乎无损"结论针对的是 **Qwen3-30B-A3B**（MoE，30B 总量）。本实验是 **1.7B 小 Dense 模型**，3-bit 权重量化对小模型更激进，精度损失天然更大。
2. **内存优化开关**：训练时为规避共享服务器 OOM，开启了 `fp16_act: true`（校准激活以 FP16 缓存）与 `low_cpu_memory: true`。前者相较论文默认的 `fp16_act: false` 可能引入轻微精度损失。
3. **bit 精度敏感**：3-bit 权重可用码点仅 8 个，对 1.7B 模型的表达能力压缩较重。
4. 以上为单次复现结果，未做超参搜索（如 `group_size`/`symmetric`/`lora_rank` 等）。

## 四、量化配置

```yaml
model: model_zoo/Qwen3-1.7B
wbits: 3
abits: 16
epochs: 60
quant_mode: lora_only      # WXA16 路径
lora_rank: 4
use_lora: true
nsamples: 128
calib_dataset: wikitext2
fp16_act: true             # 内存优化
low_cpu_memory: true       # 内存优化
```

## 五、产物位置

| 产物 | 位置 |
| --- | --- |
| 量化 checkpoint | `log/qwen3-1.7b-w3a16/slider_parameters.pth` |
| 评测结果 CSV | `log/qwen3-1.7b-w3a16/results.csv` |
| FP16 基线 | `FP16_BASELINE_qwen3-1.7b.md` |

## 六、复现命令

训练：

```bash
python main.py --config configs/qwen3-1.7b-w3a16/config.yaml \
  --output_dir log/qwen3-1.7b-w3a16 --cache_dir cache
```

评测（weight_merge）：

```bash
python main.py --config configs/qwen3-1.7b-w3a16/config.yaml \
  --output_dir log/qwen3-1.7b-w3a16 --cache_dir cache \
  --test_mode --weight_merge --eval_ppl \
  --tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande \
  --lm_eval_batch_size auto \
  --resume log/qwen3-1.7b-w3a16/slider_parameters.pth
```
