# Qwen3-1.7B W3A16 量化实验 — 进展与结果总结

> 状态：**已完成**（训练 + 评测 + 对比）　更新时间：2026-09-14

## 一、任务目标

在 Ascend NPU 上复现 SliderQuant 对 **Qwen3-1.7B** 的 **W3A16**（3-bit 权重 + FP16 激活，`lora_only` 路径）量化，评测精度并与 **FP16 全精度基线** 对比。

## 二、环境

| 项 | 值 |
| --- | --- |
| NPU 服务器 | `121.37.53.41:10001`（root），8× Ascend 910B3 |
| Docker 镜像 | `mindspeed-llm:v26.1.0-cann9.1.0-torch_npu2.7.1.post8-910b-openeuler24.03-py3.12` |
| 容器 | `sliderquant-qwen3`（挂载 NPU 1 号卡） |
| 挂载 | `/opt/zh/quant-research` → `/workspace/quant-research` |
| 依赖 | torch 2.7.1 / torch_npu 2.7.1.post8 / transformers 5.2.0 / lm-eval 0.4.13 / datasets 5.0.1 |
| 模型 | `Qwen/Qwen3-1.7B`（Dense，经 `hf-mirror.com` 下载） |

## 三、已完成工作

1. **NPU 适配**：`torch.cuda→torch.npu`、`nccl→hccl`、`nvidia-smi→npu-smi`、`import torch_npu`。
2. **transformers 5.2.0 兼容修复**（容器版本远高于 SliderQuant 预设的 4.51）：
   - `Catcher` 增加 `__getattr__` 转发（`attention_type`）
   - `apply_rotary_pos_emb` 移除 `position_ids` 参数
   - `QuantLlamaDecoderLayer.forward` 改为返回张量 + `**kwargs`/`past_key_values`/`attention_type`
   - `train_utils.py` 去掉层输出 `[0]` 索引
3. **torch 2.7 兼容**：`torch.load(..., weights_only=False)`。
4. **非 DDP 训练路径修复**：`sub_layers.train()` 分支。
5. **lm-eval 适配**：安装 0.4.13；移除失效 `ALL_TASKS` 导入、`TaskManager` 显式导入、`HFLM(device="npu")`。
6. **`low_cpu_memory` bug 修复**：`model_to_inference_mode` 跳过 `torch.nn.Identity` 层，避免清理阶段崩溃。
7. **数据准备**：wikitext2（校准/PPL）+ 6 个零样本任务。
8. **冒烟跑通**（`nsamples=16, epochs=2`）：端到端流程验证。
9. **FP16 基线评测**、**完整 W3A16 训练（epochs=60, nsamples=128）**、**W3A16 weight_merge 评测**。

## 四、实验结果

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

## 五、结论

1. W3A16 量化流程**完整跑通**（训练 → 推理模式 → PPL → 下游任务），代码在 NPU 上可用。
2. 精度有可感知下降：**PPL +5.47**，**avg-6 -6.38 个百分点**。
3. 各任务降幅分化：`arc_easy`（-12.88）、`hellaswag`（-9.36）降幅较大；`boolq`（-1.52）、`winogrande`（-2.77）相对稳健。
4. **为何不"几乎无损"**：
   - 论文"几乎无损"针对 **Qwen3-30B-A3B**（30B MoE）；本实验是 **1.7B 小 Dense 模型**，3-bit 量化更激进；
   - 为规避共享服务器 OOM，训练时开启 `fp16_act: true`（论文默认 `false`）与 `low_cpu_memory: true`，前者可能引入轻微额外损失；
   - 单次复现，未做 `group_size`/`symmetric`/`lora_rank` 等超参搜索。

## 六、训练过程要点

- **OOM 处理**：首次训练在 `Round 4` 被宿主机 OOM 杀掉（共享服务器内存压力，非代码问题）；开启 `fp16_act: true` + `low_cpu_memory: true` 后重跑成功。
- **训练耗时**：约 3.62h（单卡 NPU，28 层 × 2 阶段 × 19 窗口 × 30 epoch）。
- **显存占用**：NPU 峰值约 12.5GB（远低于 64GB 上限）。

## 七、产物位置

| 产物 | 位置 |
| --- | --- |
| 量化配置 | `configs/qwen3-1.7b-w3a16/config.yaml` |
| 量化 checkpoint | `log/qwen3-1.7b-w3a16/slider_parameters.pth` |
| 评测结果 CSV | `log/qwen3-1.7b-w3a16/results.csv` |
| FP16 基线 | `FP16_BASELINE_qwen3-1.7b.md` |
| 对比报告 | `W3A16_vs_FP16_qwen3-1.7b.md` |
| 训练日志 | `/workspace/quant-research/w3a16_train.log` |

## 八、复现命令

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
