# -*- coding: utf-8 -*-
"""SLQ 论文对标实现: GPTQ 逐层量化 + Multi-Bitwidth Shapley 敏感度 + 预测二分 TL 搜索。

对标论文 (arXiv 2605.02404, Table 1 / Section 3.3 / Algorithm 1-2):

1. 逐层量化器 = GPTQ (group_size=128, 非对称, 位宽 {2..8})。
2. 敏感度 = Multi-Bitwidth Shapley:
   对每个目标位宽 b* in B\\{bmax} 跑 P 个随机排列; 全组先置于 bmax,
   按排列顺序逐组切换到 b*, 记录每步 (EAR, KL) 边际变化;
   Shapley 值 c_m(b*) = 跨排列的平均边际贡献。
   成本 O(P*M*|B|) 次前向 (博弈独立, 可并行)。
   实现: 预计算每组在 bmax 与 b* 的 GPTQ 权重副本 (CPU), 排列内只做 copy_,
   避免重复跑 GPTQ。
3. TL 单点标定 (Algorithm 2):
   recovery ~= 1 - alpha*DKL, alpha = (1 - recovery_cal) / D_cal;
   校准比 rho = D_actual / D_predicted (锚点处测);
   预测 KL  D_hat = rho * (base_kl + sum_m c_kl[m][b_m]);
   二分找最小 avg_bits 使 D_hat <= D_thresh = (1 - target_rec) / alpha;
   guardrail: 收敛后实测 KL, 若实测/预测 相对 rho 偏离 > 2x 则拒绝 (重新锚定)。
"""
import json
import random

import numpy as np
import torch

from .gptq import gptq_quantize_weight

# 位宽预算二分用 DP (多选背包, 与论文 ILP 数学等价: 最小化 sum c subject 预算)
from .allocation import dp_allocate


def collect_activations(model, groups, inputs, device, n_rows=1024, batch=4):
    """前向 hook 收集每个量化 Linear 的输入激活 (每 Linear 采样 n_rows 行, CPU)。

    返回 {id(mod): Tensor[n_rows, in]}。在原始权重上前向执行。
    注意: hook 内只 append 到 list (不 torch.cat), 避免每批 CPU 拷贝/同步;
    每 8 批打印进度, 便于监控。
    """
    buf = {}
    handles = []
    taken = {}

    def make_hook(mid):
        def hook(mod, inp, out):
            if taken[mid] >= n_rows:
                return
            x = inp[0].detach().float()
            if x.dim() >= 3:  # [B, 1, S, H] 或 [B, S, H] -> [B*S, H]
                x = x.reshape(-1, x.shape[-1])
            take = min(n_rows - taken[mid], x.shape[0])
            if take > 0:
                buf[mid].append(x[:take].to("cpu"))
                taken[mid] += take
        return hook

    for grp in groups:
        for mod in grp["modules"]:
            mid = id(mod)
            buf[mid] = []
            taken[mid] = 0
            handles.append(mod.register_forward_hook(make_hook(mid)))

    model.eval()
    n_batches = (len(inputs) + batch - 1) // batch
    with torch.no_grad():
        for i in range(0, len(inputs), batch):
            b = torch.stack(inputs[i:i + batch]).to(device)
            model(b)
            if (i // batch) % 8 == 0:
                n_done = sum(1 for m in buf if taken[m] >= n_rows)
                print(f"[activations] batch {i // batch}/{n_batches} modules_done={n_done}/{len(buf)}", flush=True)
            if all(taken[mid] >= n_rows for mid in buf):
                break
    for h in handles:
        h.remove()
    buf = {m: torch.cat(lst, dim=0) if lst else torch.empty(0, dtype=torch.float32)
           for m, lst in buf.items()}
    missing = [m for m in buf if buf[m].shape[0] == 0]
    for m in missing:  # 兜底: 用同组第一个非空激活
        for grp in groups:
            if m in [id(mod) for mod in grp["modules"]]:
                for mod in grp["modules"]:
                    if buf[id(mod)].shape[0] > 0:
                        buf[m] = buf[id(mod)].clone()
                        break
                break
    print(f"[activations] collected {sum(b.shape[0] for b in buf.values())} rows over "
          f"{len(buf)} modules (n_rows={n_rows})")
    return buf


def gptq_group_weights(groups, m, bits, act, group_size=128, symmetric=False, damp=0.01):
    """对组 m 内每个 Linear 做 GPTQ 量化, 返回 [CPU bf16 权重列表] (不写模型)。"""
    import time
    out = []
    t0 = time.time()
    for mod in groups[m]["modules"]:
        X = act.get(id(mod))
        w = mod.weight.detach()
        dq = gptq_quantize_weight(w, X, bits, group_size, symmetric, damp, verbose=True)
        out.append(dq.detach().to("cpu"))
    print(f"  [gptq-group] m={m} name={groups[m]['name']} "
          f"mods={len(groups[m]['modules'])} done in {time.time() - t0:.1f}s", flush=True)
    return out


def write_all(groups, weights):
    """把预计算权重副本 [m][k] 写回模型 (每组多个模块)。"""
    with torch.no_grad():
        for m, ws in enumerate(weights):
            for mod, wc in zip(groups[m]["modules"], ws):
                mod.weight.copy_(wc.to(mod.weight.device).to(mod.weight.dtype))


def write_group(groups, m, weights):
    with torch.no_grad():
        for mod, wc in zip(groups[m]["modules"], weights[m]):
            mod.weight.copy_(wc.to(mod.weight.device).to(mod.weight.dtype))


def shapley_sensitivity(model, groups, act, metrics_fn, bits_list, n_perm=2,
                        group_size=128, symmetric=False, seed=2):
    """多 bitwidth Shapley 敏感度估计。

    Returns dict:
      base_ear, base_kl: 全组 bmax 状态
      c_ear/c_kl: {m: {b: shalpley 边际}}  (bmax 处为 0)
      params: list[M]
      n_perm
    """
    bmax = max(bits_list)
    M = len(groups)
    random.seed(seed)
    torch.manual_seed(seed)

    print(f"[shapley] precompute W(bmax={bmax}) for {M} groups (GPTQ)...")
    Wmax = [gptq_group_weights(groups, m, bmax, act, group_size, symmetric) for m in range(M)]
    write_all(groups, Wmax)
    base_ear, base_kl = metrics_fn()
    print(f"[shapley] base (all bmax={bmax}) EAR={base_ear:.5f} KL={base_kl:.5f}")

    sens = {m: {} for m in range(M)}   # m -> {b: [(d_ear, d_kl), ...]}
    for b in bits_list:
        if b >= bmax:
            continue
        print(f"[shapley] bitwidth {b}: precompute W(b={b}) (GPTQ)...")
        Wb = [gptq_group_weights(groups, m, b, act, group_size, symmetric) for m in range(M)]
        for p in range(n_perm):
            perm = list(range(M))
            random.shuffle(perm)
            write_all(groups, Wmax)
            prev_ear, prev_kl = base_ear, base_kl
            for m in perm:
                write_group(groups, m, Wb)
                ear, kl = metrics_fn()
                sens[m].setdefault(b, []).append((prev_ear - ear, kl - prev_kl))
                prev_ear, prev_kl = ear, kl
        print(f"[shapley] b={b}: {n_perm} permutations done")

    c_ear = {m: {b: 0.0 for b in bits_list} for m in range(M)}
    c_kl = {m: {b: 0.0 for b in bits_list} for m in range(M)}
    for m in range(M):
        for b, arr in sens[m].items():
            c_ear[m][b] = float(np.mean([a[0] for a in arr]))
            c_kl[m][b] = float(np.mean([a[1] for a in arr]))
    params = [sum(mod.weight.numel() for mod in grp["modules"]) for grp in groups]
    return {"base_ear": base_ear, "base_kl": base_kl,
            "c_ear": c_ear, "c_kl": c_kl, "params": params, "n_perm": n_perm}


def predicted_kl(sens, alloc, m2b=None):
    """Shapley 预测 KL (相对原始): base_kl + sum_m c_kl[m][b_m]。"""
    base_kl = sens["base_kl"]
    c_kl = sens["c_kl"]
    return base_kl + sum(c_kl[m].get(int(alloc[m]), 0.0) for m in range(len(alloc)))


def search_tl_predicted(sens, bits_list, target_kl, rho=1.0, tol_bits=0.02):
    """论文式预测二分: 找最小 avg_bits 使 rho*predicted_kl(alloc) <= target_kl。

    使用 DP 多选背包 (等价 ILP) 最小化预测 KL, 在预算上二分。
    """
    params = sens["params"]
    c_kl = sens["c_kl"]
    M = len(params)
    total_params = sum(params)
    costs = [[c_kl[m][b] for b in bits_list] for m in range(M)]

    def check(alloc):
        return rho * predicted_kl(sens, alloc) <= target_kl

    lo_b, hi_b = min(bits_list), max(bits_list)
    best = None
    for _ in range(30):
        mid = (lo_b + hi_b) / 2
        alloc, _ = dp_allocate(params, costs, bits_list, mid * total_params)
        if alloc is None:
            lo_b = mid
            continue
        if check(alloc):
            best = alloc
            hi_b = mid
        else:
            lo_b = mid
        if hi_b - lo_b < tol_bits:
            break
    if best is None:
        return None, None
    avg_bits = sum(b * p for b, p in zip(best, params)) / total_params
    return best, avg_bits


def apply_config_gptq(model, groups, alloc, act, group_size=128, symmetric=False):
    """按 alloc 位宽配置, 逐组 GPTQ 量化并原地写回。"""
    import time
    weights = []
    t_all = time.time()
    for m in range(len(groups)):
        weights.append(gptq_group_weights(groups, m, int(alloc[m]), act, group_size, symmetric))
        print(f"  [apply-gptq] group {m + 1}/{len(groups)} done", flush=True)
    write_all(groups, weights)
    print(f"[apply-gptq] all {len(groups)} groups quantized in {time.time() - t_all:.1f}s", flush=True)
