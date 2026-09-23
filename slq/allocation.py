# -*- coding: utf-8 -*-
"""SLQ 位宽分配: 多选背包 DP + 预算二分搜索 (论文 Section 3.3)。

问题: 给 M 个组分配位宽 b_m in B, 最小化预测度量损失 sum_m c_m(b_m),
    约束总位宽预算 sum_m params_m * b_m <= budget。
这是经典 multiple-choice knapsack, 用 DP 求解 (组数与位宽数都很小)。

DL 目标: 预测 EAR >= target_ear  (EAR 损失 = sum_m beta_m * e_m(b_m))
TL 目标: 预测 KL  <= target_kl   (KL 损失  = sum_m alpha_m * e_m(b_m))

二分搜索: 在预算上二分, 找满足质量约束的最小平均位宽。
"""
_BUDGET_UNIT = 1_000_000  # 位宽预算的整数单位: 100 万参数位


def dp_allocate(params, costs, bits_list, budget):
    """多选背包 DP: 在预算内最小化总成本。

    Args:
        params: list[M] 每组参数个数
        costs:  list[M][len(B)] 每组各位宽的预测损失
        bits_list: 位宽列表 (B)
        budget: 总位宽预算 (以 params*bits 计, 实数)
    Returns:
        (alloc: list[M] 位宽, total_cost)
    """
    M = len(params)
    cap = int(budget / _BUDGET_UNIT)
    INF = float("inf")
    dp = [[INF] * (cap + 1) for _ in range(M + 1)]
    par = [[None] * (cap + 1) for _ in range(M + 1)]
    dp[0][0] = 0.0
    for m in range(M):
        for j in range(cap + 1):
            if dp[m][j] == INF:
                continue
            for bi, b in enumerate(bits_list):
                w = int(round(params[m] * b / _BUDGET_UNIT))
                nj = j + w
                if nj > cap:
                    continue
                c = dp[m][j] + costs[m][bi]
                if c < dp[m + 1][nj]:
                    dp[m + 1][nj] = c
                    par[m + 1][nj] = (j, bi)
    feas = [j for j in range(cap + 1) if dp[M][j] != INF]
    if not feas:
        return None, INF
    best_j = min(feas, key=lambda j: dp[M][j])
    alloc = [None] * M
    j = best_j
    for m in range(M, 0, -1):
        prev_j, bi = par[m][j]
        alloc[m - 1] = bits_list[bi]
        j = prev_j
    return alloc, dp[M][best_j]


def min_bits_for_target(params, loss, bits_list, target_loss, tol_bits=0.02, verbose=True):
    """二分搜索最小平均位宽, 使预测损失 <= target_loss。

    Returns: (alloc, avg_bits) 或 (None, None)
    """
    lo_b, hi_b = min(bits_list), max(bits_list)
    best = None
    for _ in range(30):
        mid = (lo_b + hi_b) / 2
        alloc, cost = dp_allocate(params, loss, bits_list, mid * sum(params))
        if cost <= target_loss:
            best = alloc
            hi_b = mid
        else:
            lo_b = mid
        if hi_b - lo_b < tol_bits:
            break
    if best is None:
        return None, None
    avg_bits = sum(b * p for b, p in zip(best, params)) / sum(params)
    return best, avg_bits


def search_dl(params, beta, errors, bits_list, target_ear, base_ear, rho_ear,
              tol_bits=0.02, verbose=True):
    """DL 搜索: 找最小平均位宽使预测 EAR >= target_ear。

    predicted_ear(b) = base_ear - rho_ear * sum_m beta_m * e_m(b_m)
    errors: list[M] of dict {b: e_m(b)} (list-based, JSON 可序列化)
    """
    target_loss = (base_ear - target_ear) / max(rho_ear, 1e-12)
    M = len(params)
    loss = [[beta[m] * errors[m][b] for b in bits_list] for m in range(M)]
    alloc, avg_bits = min_bits_for_target(params, loss, bits_list, target_loss, tol_bits, verbose)
    return alloc, avg_bits


def search_tl(params, alpha, errors, bits_list, target_kl, rho_kl,
              tol_bits=0.02, verbose=True):
    """TL 搜索: 找最小平均位宽使预测 KL <= target_kl。

    errors: list[M] of dict {b: e_m(b)}
    """
    M = len(params)
    loss = [[alpha[m] * errors[m][b] for b in bits_list] for m in range(M)]
    alloc, avg_bits = min_bits_for_target(params, loss, bits_list, target_kl, tol_bits, verbose)
    return alloc, avg_bits
