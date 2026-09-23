# -*- coding: utf-8 -*-
"""SLQ 输出度量: EAR (Expected Acceptance Rate) 与 KL 散度 (批处理前向)。

论文定义 (Section 3.2):
    EAR = (1/N) sum_i sum_{k=1..K=10} min(p_i(k), q_i(k))
    KL  = (1/N) sum_i sum_{k in topK(p)} p_i(k) log(p_i(k)/q_i(k))
其中 p 为原始(BF16)模型在位置 i 的 next-token 分布, q 为量化模型分布,
均限制在 p 的 top-K token 上 (论文取 K=10)。

temperature: 论文在采样温度 (Qwen3 推荐 0.6) 下的分布上算 EAR/KL,
  使 top-10 质量接近 0.99 以对齐绝对阈值 EAR>=0.99。

batch_size: 批处理前向 (避免逐样本前向的低效), 大幅加速。
"""
import torch


def _to_batches(calib_inputs, batch_size):
    """list of [1, S] -> list of [B, S] (最后一批可能更小)。"""
    batches = []
    for i in range(0, len(calib_inputs), batch_size):
        chunk = calib_inputs[i:i + batch_size]
        batches.append(torch.cat(chunk, dim=0))
    return batches


@torch.no_grad()
def collect_reference(model, calib_inputs, device, topk=10, temperature=1.0, batch_size=4):
    """对原始模型跑校准集, 缓存每个样本每个位置的 top-K (索引, 概率)。"""
    ref = []
    for batch in _to_batches(calib_inputs, batch_size):
        logits = model(batch.to(device)).logits.float()  # [B, S, V]
        if temperature != 1.0:
            logits = logits / temperature
        B, S, V = logits.shape
        flat = logits.reshape(-1, V)                      # [B*S, V]
        top = flat.topk(topk, dim=-1)                     # logits 的 top-K == softmax 的 top-K
        lse = torch.logsumexp(flat, dim=-1, keepdim=True)
        p = torch.exp(top.values - lse)                   # [B*S, K]
        idx = top.indices.reshape(B, S, topk)
        p = p.reshape(B, S, topk)
        for b in range(B):
            ref.append((idx[b].to("cpu", torch.int32),
                        p[b].to("cpu", torch.float32)))
        del logits
    return ref


@torch.no_grad()
def model_metrics(model, calib_inputs, ref, device, topk=10, temperature=1.0,
                  batch_size=4, eps=1e-12):
    """对量化模型跑校准集, 返回 (EAR, KL)。ref 来自 collect_reference (同 temperature)。"""
    ear_sum = 0.0
    kl_sum = 0.0
    cnt = 0
    sample_idx = 0
    for batch in _to_batches(calib_inputs, batch_size):
        logits = model(batch.to(device)).logits.float()   # [B, S, V]
        if temperature != 1.0:
            logits = logits / temperature
        B, S, V = logits.shape
        lse = torch.logsumexp(logits, dim=-1, keepdim=True)  # [B, S, 1]
        for b in range(B):
            idx = ref[sample_idx][0].to(device).long()    # [S, K]
            p = ref[sample_idx][1].to(device).clamp(min=eps)
            sample_idx += 1
            q = torch.exp(logits[b].gather(1, idx) - lse[b])  # [S, K]
            q = q.clamp(min=eps)
            ear_sum += torch.minimum(p, q).sum().item()
            kl_sum += (p * (p / q).log()).sum().item()
            cnt += p.shape[0]
        del logits
    return ear_sum / cnt, kl_sum / cnt


@torch.no_grad()
def batched_metrics(model, input_batches, ref, device, topk=10, temperature=1.0, eps=1e-12):
    """批量版 model_metrics: input_batches 为 list of [B, S], ref 按样本展平对齐。"""
    ear_sum = 0.0
    kl_sum = 0.0
    cnt = 0
    sample_idx = 0
    for batch in input_batches:
        B, S = batch.shape
        logits = model(batch.to(device)).logits.float()  # [B, S, V]
        if temperature != 1.0:
            logits = logits / temperature
        lse = torch.logsumexp(logits, dim=-1, keepdim=True)  # [B, S, 1]
        for b in range(B):
            idx = ref[sample_idx][0].to(device).long()   # [S, K]
            p = ref[sample_idx][1].to(device).clamp(min=eps)
            sample_idx += 1
            q = torch.exp(logits[b].gather(1, idx) - lse[b])
            q = q.clamp(min=eps)
            ear_sum += torch.minimum(p, q).sum().item()
            kl_sum += (p * (p / q).log()).sum().item()
            cnt += p.shape[0]
        del logits
    return ear_sum / cnt, kl_sum / cnt
