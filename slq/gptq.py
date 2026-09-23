# -*- coding: utf-8 -*-
"""GPTQ 逐层量化器 (group-wise asymmetric, 任意位宽)。

参考 Frantar et al. 2023 的贪心误差补偿循环, 适配 SLQ 的 group-wise
非对称量化 (group_size=128)。

对单个 Linear 权重 W:[out,in] 与校准激活 X:[n,in]:
  H = 2*X^T X + damp*mean(diag)*I      # 输出 Hessian 近似 (in×in)
  Hinv = inverse(H)
  对每个 group (in 维 128 一组):
    用该 group 的 min/max 计算 scale/zero_point (per out row, 量化前固定)
    顺序遍历 group 内每一列 j:
      量化 W[:,j], 误差 err = (W[:,j] - Q(W[:,j])) / Hinv[j,j]
      把 err 按 Hinv[j, k] 传播到 group 内未量化列 k

NPU 适配: 该循环含逐列 scalar 索引 (Hinv[j,j]) 与串行小算子, 在 NPU 上
会触发海量 D2H 同步导致挂起。故 H 构造 / inverse / 逐列循环全部在 CPU
上完成 (4096 维 inverse ~1.2s, 12288 维 ~10s), 结果转回原设备。
"""
import time

import torch


@torch.no_grad()
def gptq_quantize_weight(w, X, bits, group_size=128, symmetric=False, damp=0.01,
                         verbose=False):
    """对单个 Linear 权重做 GPTQ 量化, 返回反量化权重 (与 w 同 dtype)。"""
    wf = w.detach().float()
    out, inn = wf.shape
    if bits >= 16:
        return w
    device = wf.device

    qmax = 2 ** bits - 1
    qmin = 0
    if symmetric:
        qmin_s = -(2 ** (bits - 1))
        qmax_s = 2 ** (bits - 1) - 1
    else:
        qmin_s, qmax_s = qmin, qmax

    # Hessian + 逆: 全 CPU (X 已 detach)
    xf = X.float().to("cpu")
    t0 = time.time()
    H = 2.0 * (xf.T @ xf)
    H += damp * H.diag().mean() * torch.eye(inn, device="cpu")
    Hinv = torch.inverse(H)
    if verbose:
        print(f"  [gptq] W{list(wf.shape)} rows={X.shape[0]} bits={bits} "
              f"Hessian+inverse {time.time() - t0:.1f}s", flush=True)

    W = wf.to("cpu")
    t_col = time.time()

    for g0 in range(0, inn, group_size):
        g1 = min(g0 + group_size, inn)
        sub = W[:, g0:g1]
        # 量化前固定 group 的 scale/zero_point (per out row)
        if symmetric:
            amax = sub.abs().amax(dim=-1, keepdim=True)
            scale = (amax / (2 ** (bits - 1) - 1)).clamp(min=1e-10)
            zp = torch.zeros_like(scale)
        else:
            xmin = sub.amin(dim=-1, keepdim=True)
            xmax = sub.amax(dim=-1, keepdim=True)
            scale = ((xmax - xmin) / (qmax - qmin)).clamp(min=1e-10)
            zp = torch.clamp(torch.round(qmin - xmin / scale), qmin, qmax)
        scale = scale[:, 0]
        zp = zp[:, 0]

        for j in range(g0, g1):
            w_col = W[:, j].clone()
            d = Hinv[j, j]
            if d <= 0:
                d = 1e-10
            q = torch.clamp(torch.round(w_col / scale) + zp, qmin_s, qmax_s)
            dq = (q - zp) * scale
            W[:, j] = dq
            err = (w_col - dq) / d
            # 传播到 group 内剩余列
            rem = g1 - j - 1
            if rem > 0:
                W[:, j + 1:g1] -= err.unsqueeze(1) * Hinv[j, j + 1:g1].unsqueeze(0)
        if verbose and g1 % 512 == 0:
            print(f"  [gptq] cols {g1}/{inn} {time.time() - t_col:.1f}s", flush=True)

    return W.to(device).to(w.dtype)


@torch.no_grad()
def gptq_apply_group(model, group, bits, group_size=128, symmetric=False, damp=0.01):
    """对 SLQ 分组 (多个 Linear) 做 GPTQ 量化, 使用每层各自的校准激活。

    注意: 需要每层的校准输入 X (由逐层前向缓存得到)。这里 group 内每个
    Linear 用相同的 X (近似, 对 attn/mlp 分组内各 Linear 输入不同但接近)。
    """
    raise NotImplementedError("需先建立逐层校准激活缓存 X; 见 pipeline.gptq_calibrate")
