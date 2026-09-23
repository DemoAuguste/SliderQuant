# -*- coding: utf-8 -*-
"""SLQ 静态分组量化器 (RTN)。

论文: Statistically-Lossless Quantization of Large Language Models (SLQ), COLM 2026.
P0 阶段: 静态 per-group 非对称整数量化, group_size=128, 位宽 2..8 任意, 舍入到最近 (RTN)。

设计要点:
- 权重视为 [out, in] 2D, 沿 in 维按 group_size 分组, 每组独立 scale/zero_point
- 计算在 fp32 中进行, 结果转回原 dtype (bf16/fp16), 避免低精度下的 scale 下溢
- symmetric 模式保留用于 gamma^2 方差定律的对照实验
"""
import torch


def quantize_weight_per_group(w, bits, group_size=128, symmetric=False, qmin=None, qmax=None):
    """逐组 RTN 量化, 返回与 w 同 dtype 的反量化权重。

    Args:
        w: [out, in] float tensor (fp32/fp16/bf16)
        bits: 量化位宽 (1..16)
        group_size: 分组大小 (沿 in 维)
        symmetric: 是否对称量化 (无 zero-point, 网格锚定在 0)
        qmin/qmax: 可选, 覆盖默认量化范围
    Returns:
        反量化后的权重 (与 w 同 dtype)
    """
    assert w.dim() == 2, "only support 2D linear weight"
    bits = int(bits)
    if bits >= 16:
        return w
    qmax = (2 ** bits - 1) if qmax is None else qmax
    qmin = 0 if qmin is None else qmin

    wf = w.detach().float()
    out, inn = wf.shape
    pad = (group_size - inn % group_size) % group_size
    if pad:
        w_pad = torch.cat([wf, wf.new_zeros(out, pad)], dim=1)
    else:
        w_pad = wf
    g = w_pad.reshape(out, -1, group_size)  # [out, n_groups, gs]

    if symmetric:
        amax = g.abs().amax(dim=-1, keepdim=True)
        scale = amax / (2 ** (bits - 1) - 1)
        scale = scale.clamp(min=1e-10)
        z = torch.zeros_like(scale)
    else:
        xmin = g.amin(dim=-1, keepdim=True)
        xmax = g.amax(dim=-1, keepdim=True)
        scale = (xmax - xmin) / qmax
        scale = scale.clamp(min=1e-10)
        z = torch.clamp(torch.round(-xmin / scale), qmin, qmax)

    q = torch.clamp(torch.round(g / scale) + z, qmin, qmax)
    dq = (q - z) * scale
    dq = dq.reshape(out, -1)
    if pad:
        dq = dq[:, :inn]
    return dq.to(w.dtype)


@torch.no_grad()
def quantize_linear_inplace(linear, bits, group_size=128, symmetric=False):
    """原地量化一个 nn.Linear 的权重。返回原权重副本 (用于恢复)。"""
    orig = linear.weight.detach().clone()
    linear.weight.copy_(quantize_weight_per_group(linear.weight, bits, group_size, symmetric))
    return orig
