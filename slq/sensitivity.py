# -*- coding: utf-8 -*-
"""SLQ 敏感度估计 (Linear 方法, 论文 Section 3.3 / Appendix A.2)。

论文的线性估计:
    Delta_KL(b) ~= sum_m alpha_m * e(b)_m
    Delta_EAR(b) ~= sum_m beta_m * e(b)_m
其中 e(b)_m = (1/|G_m|) sum_{l in G_m} ||W_l - Q(W_l, b)||_F^2 / ||W_l||_F^2
为归一化重建误差, alpha/beta 通过对每个组注入噪声 (量化到锚点位宽) 用 O(M) 次
前向估计。

P0 实现:
- weight_errors: 纯张量运算计算全表 e(b)_m (无需前向)
- estimate_coeffs: 每个组单独量化到 anchor_bits (其余组保持 bmax), 跑一次校准集,
  得该组的边际 EAR/KL 损失, 除以 e(anchor)_m 得 beta_m/alpha_m (O(M) 次前向)
- 另测一次"全组 anchor"配置用于校准比 rho (论文 TL 中的 calibration ratio 思路,
  用于修正线性模型对互作用的低估/高估)
"""
from .quantizer import quantize_weight_per_group


def weight_errors(groups, bits_list, group_size=128, symmetric=False):
    """e(b)_m 全表。返回 dict {(m, b): float} 与 dict {m: params}。"""
    errors = {}
    params = {}
    for m, grp in enumerate(groups):
        num = {b: 0.0 for b in bits_list}
        den = 0.0
        n_params = 0
        for mod in grp["modules"]:
            w = mod.weight.detach().float()
            n_params += w.numel()
            den += w.square().sum().item()
            for b in bits_list:
                dq = quantize_weight_per_group(w, b, group_size, symmetric)
                num[b] += (w - dq).square().sum().item()
        den = max(den, 1e-12)
        for b in bits_list:
            errors[(m, b)] = num[b] / den
        params[m] = n_params
    return errors, params


def estimate_coeffs(model, groups, metrics_fn, bits_list, anchor_bits, group_size=128,
                    symmetric=False, device=None):
    """O(M) 次前向估计每个组的边际 KL/EAR 系数与校准比 rho。

    Args:
        metrics_fn: callable() -> (ear, kl)  对当前权重状态跑校准集
    Returns:
        coeff: dict {m: {'beta': .., 'alpha': ..}}  (EAR 损失/unit-error, KL 损失/unit-error)
        base_metrics: (ear, kl) 在全部 bmax 配置下的度量
        anchor_metrics: (ear, kl) 在全部 anchor 配置下的度量
        rho_ear: (base_ear - anchor_ear) / sum_m beta_m * e(anchor)_m
        rho_kl:  (anchor_kl - base_kl)   / sum_m alpha_m * e(anchor)_m
    """
    bmax = max(bits_list)
    # ---- 先算权重误差表 (在原始权重上, 无前向) ----
    errors, params = weight_errors(groups, bits_list, group_size, symmetric)
    e_anchor = {m: errors[(m, anchor_bits)] for m in range(len(groups))}

    # ---- 全部 bmax 基线 ----
    for grp in groups:
        _apply_group(grp, bmax, group_size, symmetric)
    base_ear, base_kl = metrics_fn()
    print(f"[sensitivity] base (all b={bmax}) EAR={base_ear:.5f} KL={base_kl:.5f}")

    # ---- 每组合并到 anchor, 测量边际 ----
    coeff = {}
    marg_loss_ear = 0.0
    marg_loss_kl = 0.0
    for m, grp in enumerate(groups):
        _apply_group(grp, anchor_bits, group_size, symmetric)
        ear, kl = metrics_fn()
        d_ear = base_ear - ear
        d_kl = kl - base_kl
        coeff[m] = {
            "beta": d_ear / max(e_anchor[m], 1e-12),
            "alpha": d_kl / max(e_anchor[m], 1e-12),
            "e_anchor": e_anchor[m],
            "marg_ear": d_ear,
            "marg_kl": d_kl,
        }
        marg_loss_ear += d_ear
        marg_loss_kl += d_kl
        _apply_group(grp, bmax, group_size, symmetric)  # 恢复
        print(f"[sensitivity] group {m} {grp['name']}: dEAR={d_ear:+.5f} dKL={d_kl:+.5f} "
              f"e({anchor_bits})={e_anchor[m]:.5f}")

    # ---- 全部 anchor 配置, 校准 rho ----
    for grp in groups:
        _apply_group(grp, anchor_bits, group_size, symmetric)
    anchor_ear, anchor_kl = metrics_fn()
    for grp in groups:
        _apply_group(grp, bmax, group_size, symmetric)
    pred_ear_loss = marg_loss_ear
    pred_kl_loss = marg_loss_kl
    rho_ear = (base_ear - anchor_ear) / max(pred_ear_loss, 1e-12)
    rho_kl = (anchor_kl - base_kl) / max(pred_kl_loss, 1e-12)
    print(f"[sensitivity] all-anchor EAR={anchor_ear:.5f} KL={anchor_kl:.5f} "
          f"rho_ear={rho_ear:.3f} rho_kl={rho_kl:.3f}")
    return (coeff, (base_ear, base_kl), (anchor_ear, anchor_kl), rho_ear, rho_kl,
            errors, params)


def _apply_group(grp, bits, group_size=128, symmetric=False):
    """将组内所有 Linear 量化到 bits (原地)。"""
    from .quantizer import quantize_weight_per_group
    with torch_no_grad():
        for mod in grp["modules"]:
            w = mod.weight.detach().float()
            dq = quantize_weight_per_group(w, bits, group_size, symmetric).to(mod.weight.dtype)
            mod.weight.copy_(dq)


import torch
torch_no_grad = torch.no_grad
