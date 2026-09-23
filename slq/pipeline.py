# -*- coding: utf-8 -*-
"""SLQ 管线编排: 模型加载 / 组构建 / 位宽应用 / DL-TL 搜索 / 评测。"""
import copy
import json
import os

import torch
import torch.nn as nn

from .quantizer import quantize_weight_per_group
from .metrics import collect_reference, model_metrics
from .sensitivity import weight_errors, estimate_coeffs
from .allocation import dp_allocate, search_dl, search_tl


def get_device():
    if torch.npu.is_available():
        return torch.device("npu:0")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path, use_bfloat16=True, device=None, trust_remote_code=True):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    if device is None:
        device = get_device()
    dtype = torch.bfloat16 if use_bfloat16 else torch.float16
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, torch_dtype=dtype,
        low_cpu_mem_usage=True, trust_remote_code=trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, legacy=False)
    model.to(device).eval()
    model.config.use_cache = False
    return model, tokenizer, device


def get_decoder_layers(model):
    m = model
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return m.model.layers
    if hasattr(m, "layers"):
        return m.layers
    raise ValueError("cannot locate decoder layers")


def build_groups(model, group_mode="split"):
    """构建量化组。

    group_mode:
      - split: 每层两组: attn={q,k,v,o}, mlp={gate,up,down}
      - layer: 每层一组 (该层所有 Linear)
      - all:   每组单个 Linear (最细粒度)
    """
    layers = get_decoder_layers(model)
    groups = []
    for li, layer in enumerate(layers):
        if group_mode == "split":
            attn_mods, mlp_mods = [], []
            for name, mod in layer.named_modules():
                if isinstance(mod, nn.Linear):
                    if "self_attn" in name:
                        attn_mods.append(mod)
                    elif "mlp" in name:
                        mlp_mods.append(mod)
            if attn_mods:
                groups.append({"name": f"L{li}.attn", "modules": attn_mods})
            if mlp_mods:
                groups.append({"name": f"L{li}.mlp", "modules": mlp_mods})
        elif group_mode == "layer":
            mods = [mod for name, mod in layer.named_modules() if isinstance(mod, nn.Linear)]
            groups.append({"name": f"L{li}", "modules": mods})
        elif group_mode == "all":
            for name, mod in layer.named_modules():
                if isinstance(mod, nn.Linear):
                    groups.append({"name": f"L{li}.{name}", "modules": [mod]})
        else:
            raise ValueError(f"unknown group_mode {group_mode}")
    return groups


@torch.no_grad()
def apply_config(model, groups, bit_config, group_size=128, symmetric=False):
    """按 per-group 位宽配置量化模型权重 (原地)。"""
    for m, grp in enumerate(groups):
        bits = int(bit_config[m])
        for mod in grp["modules"]:
            w = mod.weight.detach().float()
            dq = quantize_weight_per_group(w, bits, group_size, symmetric).to(mod.weight.dtype)
            mod.weight.copy_(dq)


@torch.no_grad()
def apply_uniform(model, groups, bits, group_size=128, symmetric=False):
    for grp in groups:
        for mod in grp["modules"]:
            w = mod.weight.detach().float()
            dq = quantize_weight_per_group(w, bits, group_size, symmetric).to(mod.weight.dtype)
            mod.weight.copy_(dq)


def describe_config(groups, bit_config):
    n_quant = 0
    per_bits = {}
    for m, grp in enumerate(groups):
        b = int(bit_config[m])
        n_params = sum(mod.weight.numel() for mod in grp["modules"])
        n_quant += n_params
        per_bits.setdefault(b, 0)
        per_bits[b] += n_params
    avg = sum(int(bit_config[m]) * sum(mod.weight.numel() for mod in groups[m]["modules"])
              for m in range(len(groups))) / n_quant
    return {"avg_bits": round(avg, 3), "per_bits": {k: v for k, v in sorted(per_bits.items())},
            "n_params": n_quant}


def run_dl_search(model, groups, calib_inputs, ref, device, cfg, sens_data=None):
    """DL 搜索: 敏感度 -> 分配 -> 二分 -> 验证。返回结果 dict。

    sens_data: 预计算的敏感度数据库 (dict), 用于断点续跑, 跳过 estimate_coeffs。
    """
    bits_list = cfg["bits_list"]
    group_size = cfg["group_size"]
    symmetric = cfg.get("symmetric", False)
    anchor = cfg.get("anchor_bits", min(4, max(bits_list)))
    target_ear = cfg["dl_target_ear"]
    dl_margin = cfg.get("dl_relative_margin", 0.01)  # 相对 base EAR 的可接受损失
    topk = cfg.get("topk", 10)
    temperature = cfg.get("temperature", 1.0)
    sens_n = cfg.get("sens_nsamples", len(calib_inputs))

    sens_inputs = calib_inputs[:sens_n]
    sens_ref = ref[:sens_n]
    metrics_fn = lambda: model_metrics(model, sens_inputs, sens_ref, device, topk=topk,
                                       temperature=temperature)

    # 原始权重快照: 必须在任何敏感度/量化前取, 否则快照到的是量化态权重
    orig = _snapshot(groups)

    if sens_data is None:
        print(f"=== [DL] sensitivity estimation (sens_nsamples={sens_n}) ===")
        _restore(groups, orig)  # 保证敏感度从原始权重启动
        (coeff, (base_ear, base_kl), (anchor_ear, anchor_kl), rho_ear, rho_kl,
         errors, params) = estimate_coeffs(
            model, groups, metrics_fn, bits_list, anchor, group_size, symmetric, device)
        M = len(groups)
        sens_data = {
            "coeff": [coeff[m] for m in range(M)],
            "errors": [{b: errors[(m, b)] for b in bits_list} for m in range(M)],
            "params": [params[m] for m in range(M)],
            "base_ear": base_ear, "base_kl": base_kl,
            "anchor_ear": anchor_ear, "anchor_kl": anchor_kl,
            "rho_ear": rho_ear, "rho_kl": rho_kl,
        }
    else:
        print("=== [DL] reuse sensitivity cache ===")
    coeff = {m: sens_data["coeff"][m] for m in range(len(sens_data["coeff"]))}
    # JSON 往返后内层 dict 键变字符串, 转回 int
    errors = [{int(k): v for k, v in err_dict.items()} for err_dict in sens_data["errors"]]
    params = sens_data["params"]           # list[float]
    base_ear = sens_data["base_ear"]
    base_kl = sens_data["base_kl"]
    anchor_ear = sens_data["anchor_ear"]
    anchor_kl = sens_data["anchor_kl"]
    rho_ear = sens_data["rho_ear"]
    rho_kl = sens_data["rho_kl"]

    beta = {m: coeff[m]["beta"] for m in coeff}

    print("=== [DL] bitwidth search (verify-and-adjust) ===")
    # 绝对目标 EAR 受 base top-10 质量上限约束 (小模型 top-10 质量<0.99),
    # 取 min(绝对目标, base_ear - 相对裕度)
    eff_target = min(target_ear, base_ear - dl_margin)
    print(f"[DL] target_ear={target_ear} base_ear={base_ear:.5f} -> eff_target={eff_target:.5f}")

    # 分配排序成本: 只用于"预算内如何分", 目标达标靠实测二分保证
    loss_ear = [[beta[m] * errors[m][b] for b in bits_list] for m in range(len(params))]
    total_params = sum(params)

    def _measure(alloc, use_sens=True):
        _restore(groups, orig)
        apply_config(model, groups, alloc, group_size, symmetric)
        if use_sens:
            return model_metrics(model, sens_inputs, sens_ref, device, topk=topk,
                                 temperature=temperature)
        return model_metrics(model, calib_inputs, ref, device, topk=topk,
                             temperature=temperature)

    lo_b, hi_b = min(bits_list), max(bits_list)
    best = None
    for _ in range(14):
        mid = (lo_b + hi_b) / 2
        alloc, _ = dp_allocate(params, loss_ear, bits_list, mid * total_params)
        if alloc is None:  # 预算过小不可行
            lo_b = mid
            continue
        ear, _ = _measure(alloc, use_sens=True)
        print(f"[DL] budget {mid:.2f} avg_bits={sum(b*p for b, p in zip(alloc, params)) / total_params:.3f} "
              f"EAR(sens)={ear:.5f}")
        if ear >= eff_target:
            best = alloc
            hi_b = mid
        else:
            lo_b = mid
        if hi_b - lo_b < 0.03:
            break
    if best is None:
        raise RuntimeError("DL search failed: cannot meet EAR target even at max bits")

    ear, kl = _measure(best, use_sens=False)  # 全量校准集最终验证
    avg_bits = sum(b * p for b, p in zip(best, params)) / total_params
    desc = describe_config(groups, best)
    print(f"=== [DL] result: avg_bits={desc['avg_bits']} EAR={ear:.5f} KL={kl:.5f} "
          f"(target EAR>={eff_target:.5f}) ===")
    print(f"    per-bit distribution: {desc['per_bits']}")
    return {"mode": "dl", "avg_bits": desc["avg_bits"], "ear": ear, "kl": kl,
            "target_ear": target_ear, "eff_target_ear": eff_target, "alloc": best,
            "per_bits": desc["per_bits"], "base_ear": base_ear, "anchor_ear": anchor_ear,
            "rho_ear": rho_ear, "sens": sens_data}


def _snapshot(groups):
    return [m.weight.detach().clone() for grp in groups for m in grp["modules"]]


def _restore(groups, snapshot):
    with torch.no_grad():
        i = 0
        for grp in groups:
            for mod in grp["modules"]:
                mod.weight.copy_(snapshot[i])
                i += 1


def run_tl_search(model, groups, calib_inputs, ref, device, cfg, anchor_recovery, kl_cal,
                  bf16_scores=None, sens_data=None):
    """TL 搜索: 单点标定 + verify-and-adjust 二分。

    anchor_recovery: 标定配置的 benchmark 恢复率 (relative to BF16)
    kl_cal: 标定配置相对原始模型的 KL (同一配置下测得)
    """
    bits_list = cfg["bits_list"]
    group_size = cfg["group_size"]
    symmetric = cfg.get("symmetric", False)
    anchor = cfg.get("anchor_bits", min(4, max(bits_list)))
    target_rec = cfg["tl_target_recovery"]
    topk = cfg.get("topk", 10)
    temperature = cfg.get("temperature", 1.0)
    sens_n = cfg.get("sens_nsamples", len(calib_inputs))

    sens_inputs = calib_inputs[:sens_n]
    sens_ref = ref[:sens_n]
    metrics_fn = lambda: model_metrics(model, sens_inputs, sens_ref, device, topk=topk,
                                       temperature=temperature)

    # 原始权重快照: 必须在任何敏感度/量化前取
    orig = _snapshot(groups)

    if sens_data is None:
        print(f"=== [TL] sensitivity estimation (sens_nsamples={sens_n}) ===")
        _restore(groups, orig)  # 保证敏感度从原始权重启动
        (coeff, (base_ear, base_kl), (anchor_ear, anchor_kl), rho_ear, rho_kl,
         errors_t, params_t) = estimate_coeffs(
            model, groups, metrics_fn, bits_list, anchor, group_size, symmetric, device)
        M = len(groups)
        sens_data = {
            "coeff": [coeff[m] for m in range(M)],
            "errors": [{b: errors_t[(m, b)] for b in bits_list} for m in range(M)],
            "params": [params_t[m] for m in range(M)],
            "base_ear": base_ear, "base_kl": base_kl,
            "anchor_ear": anchor_ear, "anchor_kl": anchor_kl,
            "rho_ear": rho_ear, "rho_kl": rho_kl,
        }
    else:
        print("=== [TL] reuse sensitivity cache ===")
    M = len(sens_data["coeff"])
    coeff = {m: sens_data["coeff"][m] for m in range(M)}
    errors = [{int(k): v for k, v in err_dict.items()} for err_dict in sens_data["errors"]]
    params = sens_data["params"]
    base_ear = sens_data["base_ear"]
    base_kl = sens_data["base_kl"]
    alpha = {m: coeff[m]["alpha"] for m in coeff}

    # 单点标定: recovery ≈ 1 - slope * DKL  (论文 Algorithm 2)
    d_kl_cal = max(kl_cal - base_kl, 1e-6)
    slope = (1.0 - anchor_recovery) / d_kl_cal
    target_kl = (1.0 - target_rec) / slope
    print(f"=== [TL] single-point: recovery_cal={anchor_recovery:.4f} DKL_cal={d_kl_cal:.5f} "
          f"slope={slope:.3f} target_kl={target_kl:.5f} (target recovery>={target_rec}) ===")

    # 分配排序成本 (KL), 达标由实测二分保证
    loss_kl = [[alpha[m] * errors[m][b] for b in bits_list] for m in range(M)]
    total_params = sum(params)

    def _measure(alloc, use_sens=True):
        _restore(groups, orig)
        apply_config(model, groups, alloc, group_size, symmetric)
        if use_sens:
            return model_metrics(model, sens_inputs, sens_ref, device, topk=topk,
                                 temperature=temperature)
        return model_metrics(model, calib_inputs, ref, device, topk=topk,
                             temperature=temperature)

    lo_b, hi_b = min(bits_list), max(bits_list)
    best = None
    for _ in range(14):
        mid = (lo_b + hi_b) / 2
        alloc, _ = dp_allocate(params, loss_kl, bits_list, mid * total_params)
        if alloc is None:
            lo_b = mid
            continue
        _, kl = _measure(alloc, use_sens=True)
        print(f"[TL] budget {mid:.2f} avg_bits={sum(b*p for b, p in zip(alloc, params)) / total_params:.3f} "
              f"KL(sens)={kl:.5f}")
        if kl - base_kl <= target_kl:
            best = alloc
            hi_b = mid
        else:
            lo_b = mid
        if hi_b - lo_b < 0.03:
            break
    if best is None:
        raise RuntimeError("TL search failed")

    ear, kl = _measure(best, use_sens=False)  # 全量校准集最终验证
    avg_bits = sum(b * p for b, p in zip(best, params)) / total_params
    desc = describe_config(groups, best)
    print(f"=== [TL] result: avg_bits={desc['avg_bits']} EAR={ear:.5f} KL={kl:.5f} ===")
    print(f"    per-bit distribution: {desc['per_bits']}")
    return {"mode": "tl", "avg_bits": desc["avg_bits"], "ear": ear, "kl": kl,
            "target_kl": target_kl, "alloc": best, "per_bits": desc["per_bits"],
            "anchor_recovery": anchor_recovery, "slope": slope, "kl_cal": kl_cal,
            "bf16_scores": bf16_scores, "sens": sens_data}
