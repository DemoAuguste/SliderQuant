# Qwen3-1.7B 全精度评测基线（FP16/BF16）

- 评测日期：2026-09-14
- 模型：`Qwen/Qwen3-1.7B`（Dense，约 1.7B 参数）
- 精度：BF16（全精度基线，未量化）
- 硬件：Ascend 910B3（NPU，8 卡）
- 运行环境：MindSpeed-LLM 容器 `v26.1.0-cann9.1.0-torch_npu2.7.1.post8-910b-openeuler24.03-py3.12`
- 依赖版本：torch 2.7.1 / torch_npu 2.7.1.post8 / transformers 5.2.0 / lm-eval 0.4.13

## 数据集

| 用途 | 数据集 | 说明 |
| --- | --- | --- |
| 校准（量化训练） | wikitext2 | `Salesforce/wikitext`，train 集，128 样本，seqlen 2048 |
| PPL 评测 | wikitext2 | test 集 |
| 零样本常识推理 | piqa / arc_easy / arc_challenge / boolq / hellaswag / winogrande | 经 lm-eval 由 `hf-mirror.com` 下载 |

## 评测结果

### 语言模型困惑度（PPL，越低越好）

| 数据集 | PPL |
| --- | --- |
| wikitext2 | 16.7122 |

### 零样本常识推理（准确率，越高越好）

| 任务 | 准确率 |
| --- | --- |
| piqa | 72.36% |
| arc_easy | 69.57% |
| arc_challenge | 43.34% |
| boolq | 77.61% |
| hellaswag | 60.36% |
| winogrande | 61.25% |
| **avg-5** | **61.38%** |
| **avg-6** | **64.08%** |

## 复现命令

```bash
python main.py \
  --config configs/qwen3-1.7b-w3a16/config.yaml \
  --output_dir log/qwen3-1.7b-fp16-baseline \
  --cache_dir cache \
  --wbits 16 --abits 16 \
  --eval_ppl \
  --tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande \
  --lm_eval_batch_size auto
```

## 备注

- 评测时 NPU 出现瞬时 OOM 告警（`auto` 批量探测阶段），lm-eval 自动回退到 `batch_size=32` 后正常完成，结果有效。
- c4 未纳入本次基线（`datautils.py` 硬编码本地路径 `/SliderQuant/datasets_local/c4`，需单独准备后可补充）。
- 该基线用于与后续 Qwen3-1.7B W3A16 量化结果对比。