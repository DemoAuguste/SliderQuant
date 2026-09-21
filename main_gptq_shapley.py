# -*- coding: utf-8 -*-
"""SLQ 论文对标 TL 复现入口: GPTQ + Shapley + 单点标定。

用法:
    python main_gptq_shapley.py --config configs/qwen3-8b-slq-tl.yaml --mode tl
"""
import argparse
import json
import os

import torch

from main_slq import (eval_lm_eval, load_calibration, restore_weights,
                      snapshot_weights)
from slq.gptq_shapley import (apply_config_gptq, collect_activations,
                              search_tl_predicted, shapley_sensitivity,
                              write_all)
from slq.metrics import collect_reference, model_metrics
from slq.pipeline import build_groups, describe_config, get_device, load_model


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", default="tl", choices=["tl", "eval"])
    ap.add_argument("--output_dir", default="log/slq")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--n_perm", type=int, default=2, help="Shapley 随机排列数 P")
    ap.add_argument("--sens_nsamples", type=int, default=None)
    ap.add_argument("--n_gptq_rows", type=int, default=1024, help="每 Linear GPTQ 校准行数")
    ap.add_argument("--apply_config", default=None, help="eval 模式位宽 json")
    ap.add_argument("--skip_bench", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        raw = f.read()
    try:
        cfg = json.loads(raw)
    except json.JSONDecodeError:
        cfg = yaml.safe_load(raw)
    run_name = args.run_name or os.path.splitext(os.path.basename(args.config))[0]
    out_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    device = get_device()
    print(f"[GQ] device={device} config={args.config} mode={args.mode}")

    model, tokenizer, device = load_model(cfg["model"], cfg.get("use_bfloat16", True), device)
    groups = build_groups(model, cfg.get("group_mode", "split"))
    print(f"[GQ] groups: {len(groups)}")
    inputs = load_calibration(args, cfg, cfg["model"])
    topk = cfg.get("topk", 10)
    temperature = cfg.get("temperature", 1.0)
    ref = collect_reference(model, inputs, device, topk=topk, temperature=temperature)
    print("[GQ] reference cache built")
    snapshot = snapshot_weights(groups)
    bits_list = cfg["bits_list"]
    group_size = cfg.get("group_size", 128)
    symmetric = cfg.get("symmetric", False)
    calib_bits = cfg.get("tl_calib_bits", 6)
    target_rec = cfg["tl_target_recovery"]
    tasks = cfg.get("eval_tasks", "piqa,arc_easy")
    sens_n = args.sens_nsamples or cfg.get("sens_nsamples", len(inputs))
    n_perm = args.n_perm

    sens_inputs = inputs[:sens_n]
    sens_ref = ref[:sens_n]
    metrics_fn = lambda: model_metrics(model, sens_inputs, sens_ref, device, topk=topk,
                                       temperature=temperature)

    print(f"[GQ] collect GPTQ activations (n_rows={args.n_gptq_rows}, 128 samples) on original weights...")
    act = collect_activations(model, groups, inputs[:128], device,
                              n_rows=args.n_gptq_rows, batch=8)

    result = {"config": cfg, "n_groups": len(groups), "n_perm": n_perm,
              "sens_nsamples": sens_n}

    # ---------- BF16 基线 (缓存) ----------
    bf16_cache = os.path.join(out_dir, "gq_bf16_scores.json")
    if os.path.exists(bf16_cache):
        with open(bf16_cache, "r", encoding="utf-8") as f:
            bf16_scores = json.load(f)
        print(f"[GQ] load BF16 scores from {bf16_cache}")
    else:
        bf16_scores = eval_lm_eval(model, tokenizer, tasks, cfg.get("lm_eval_batch_size", "8"), device)
        with open(bf16_cache, "w", encoding="utf-8") as f:
            json.dump(bf16_scores, f, ensure_ascii=False, indent=2)
    bf16_avg = sum(bf16_scores.values()) / len(bf16_scores)
    print(f"[GQ] BF16 avg-{len(bf16_scores)}: {bf16_avg:.5f}")

    # ---------- 锚点: uniform-calib_bits (GPTQ) ----------
    print(f"[GQ] anchor: uniform-{calib_bits} (GPTQ), KL + benchmark...")
    apply_config_gptq(model, groups, [calib_bits] * len(groups), act, group_size, symmetric)
    _, kl_cal = model_metrics(model, sens_inputs, sens_ref, device, topk=topk,
                              temperature=temperature)
    anchor_cache = os.path.join(out_dir, f"gq_anchor_{calib_bits}b_scores.json")
    if os.path.exists(anchor_cache):
        with open(anchor_cache, "r", encoding="utf-8") as f:
            anchor_scores = json.load(f)
        print(f"[GQ] load anchor scores from {anchor_cache}")
    else:
        anchor_scores = eval_lm_eval(model, tokenizer, tasks, cfg.get("lm_eval_batch_size", "8"), device)
        with open(anchor_cache, "w", encoding="utf-8") as f:
            json.dump(anchor_scores, f, ensure_ascii=False, indent=2)
    anchor_avg = sum(anchor_scores.values()) / len(anchor_scores)
    anchor_rec = anchor_avg / bf16_avg
    print(f"[GQ] anchor uniform-{calib_bits}: KL={kl_cal:.5f} avg={anchor_avg:.5f} "
          f"recovery={anchor_rec:.5f}")
    result["kl_cal"] = kl_cal
    result["anchor_recovery"] = anchor_rec
    result["bf16_avg"] = bf16_avg
    result["bf16_scores"] = bf16_scores
    result["anchor_scores"] = anchor_scores

    # ---------- 恢复原始权重, 然后 Shapley 敏感度 ----------
    restore_weights(groups, snapshot)
    print(f"[GQ] shapley sensitivity (P={n_perm}, sens_n={sens_n}, bits={bits_list})...")
    sens = shapley_sensitivity(model, groups, act, metrics_fn, bits_list, n_perm=n_perm,
                               group_size=group_size, symmetric=symmetric,
                               seed=cfg.get("seed", 2))
    result["sens_meta"] = {"base_ear": sens["base_ear"], "base_kl": sens["base_kl"],
                           "params": sens["params"], "n_perm": sens["n_perm"]}
    sens_json = os.path.join(out_dir, "gq_sens.json")
    with open(sens_json, "w", encoding="utf-8") as f:
        json.dump({"base_ear": sens["base_ear"], "base_kl": sens["base_kl"],
                   "c_ear": {str(m): {str(b): v for b, v in cm.items()} for m, cm in sens["c_ear"].items()},
                   "c_kl": {str(m): {str(b): v for b, v in cm.items()} for m, cm in sens["c_kl"].items()},
                   "params": sens["params"], "n_perm": sens["n_perm"]},
                  f, ensure_ascii=False, indent=2)
    print(f"[GQ] sensitivity saved to {sens_json}")

    # ---------- TL 单点标定 + 预测二分 (论文 Algorithm 2) ----------
    alpha = (1.0 - anchor_rec) / max(kl_cal, 1e-6)
    d_thresh = (1.0 - target_rec) / alpha
    from slq.gptq_shapley import predicted_kl
    d_pred_anchor = predicted_kl(sens, [calib_bits] * len(groups))
    rho = kl_cal / max(d_pred_anchor, 1e-8)
    print(f"[GQ] single-point: alpha={alpha:.4f} D_thresh={d_thresh:.5f} "
          f"D_pred_anchor={d_pred_anchor:.5f} rho={rho:.4f}")
    alloc, avg_bits = search_tl_predicted(sens, bits_list, d_thresh, rho)
    if alloc is None:
        raise RuntimeError("TL search failed (no feasible config)")
    desc = describe_config(groups, alloc)
    print(f"[GQ] predicted alloc: avg_bits={desc['avg_bits']} per_bits={desc['per_bits']}")
    result["avg_bits"] = desc["avg_bits"]
    result["alloc"] = alloc
    result["per_bits"] = desc["per_bits"]
    result["alpha"] = alpha
    result["d_thresh"] = d_thresh
    result["rho"] = rho
    result["d_pred_anchor"] = d_pred_anchor
    result["predicted_kl"] = predicted_kl(sens, alloc)

    # ---------- guardrail: 实测 KL (论文: 实测/预测 相对 rho 偏离 >2x 则拒绝) ----------
    restore_weights(groups, snapshot)
    apply_config_gptq(model, groups, alloc, act, group_size, symmetric)
    _, kl_actual = model_metrics(model, inputs, ref, device, topk=topk, temperature=temperature)
    d_pred = predicted_kl(sens, alloc)
    rho_actual = kl_actual / max(d_pred, 1e-8)
    ratio = rho_actual / max(rho, 1e-8)
    print(f"[GQ] guardrail: actual_kl={kl_actual:.5f} predicted_kl={d_pred:.5f} "
          f"rho_actual={rho_actual:.4f} ratio_vs_anchor_rho={ratio:.3f}")
    result["kl_actual"] = kl_actual
    result["rho_actual"] = rho_actual
    if ratio > 2.0 or ratio < 0.5:
        print("[GQ] WARNING: guardrail violated (ratio >2x), re-anchoring needed")
        result["guardrail"] = "violated"
    else:
        result["guardrail"] = "ok"

    # ---------- 最终 benchmark ----------
    if not args.skip_bench:
        scores = eval_lm_eval(model, tokenizer, tasks, cfg.get("lm_eval_batch_size", "8"), device)
        avg = sum(scores.values()) / len(scores)
        rec = avg / bf16_avg
        result["scores"] = scores
        result["avg_score"] = avg
        result["recovery"] = rec
        print(f"[GQ] TL eval: avg={avg:.5f} recovery={rec:.5f} (target>={target_rec})")
        with open(os.path.join(out_dir, "gq_slq_tl.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    else:
        print("[GQ] --skip_bench: 跳过最终评测")
        with open(os.path.join(out_dir, "gq_slq_tl.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
