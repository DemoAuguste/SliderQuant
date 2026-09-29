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
    ap.add_argument("--n_groups", type=int, default=None,
                    help="只取前 N 组用于端到端快速验证 (None=全部)")
    ap.add_argument("--apply_config", default=None, help="eval 模式位宽 json")
    ap.add_argument("--skip_bench", action="store_true")
    ap.add_argument("--sens_json", default=None,
                    help="复用已保存的 shapley sensitivity (gq_sens.json), 跳过预计算")
    ap.add_argument("--resume_values", default=None,
                    help="复用已有 gq_slq_tl.json 的 bf16/anchor 数值, 跳过基线阶段")
    ap.add_argument("--thresh_scale", type=float, default=1.0,
                    help="D_thresh 安全系数 (<1 更保守, 补偿线性恢复模型在低位宽的低估)")
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
    gptq_cache_dir = os.path.join(out_dir, "gptq_cache")
    device = get_device()
    print(f"[GQ] device={device} config={args.config} mode={args.mode}")

    model, tokenizer, device = load_model(cfg["model"], cfg.get("use_bfloat16", True), device)
    groups = build_groups(model, cfg.get("group_mode", "split"))
    if args.n_groups is not None:
        groups = groups[:args.n_groups]
        print(f"[GQ] ** E2E-FAST: only using first {len(groups)} groups for validation**")
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

    # ---------- 可选: 从上次结果复用 bf16/anchor 数值 ----------
    resumed = {}
    if args.resume_values:
        with open(args.resume_values, "r", encoding="utf-8") as f:
            resumed = json.load(f)
        need = ["bf16_avg", "bf16_scores", "kl_cal", "anchor_recovery"]
        miss = [k for k in need if k not in resumed]
        if miss:
            raise RuntimeError(f"resume_values 缺少字段: {miss}")
        print(f"[GQ] resume bf16/anchor values from {args.resume_values}")

    # ---------- BF16 基线 (缓存) ----------
    if resumed:
        bf16_scores = resumed["bf16_scores"]
        print(f"[GQ] load BF16 scores from resume json")
    elif os.path.exists(os.path.join(out_dir, "gq_bf16_scores.json")):
        with open(os.path.join(out_dir, "gq_bf16_scores.json"), "r", encoding="utf-8") as f:
            bf16_scores = json.load(f)
        print(f"[GQ] load BF16 scores from cache")
    else:
        bf16_scores = eval_lm_eval(model, tokenizer, tasks, cfg.get("lm_eval_batch_size", "8"), device)
        with open(os.path.join(out_dir, "gq_bf16_scores.json"), "w", encoding="utf-8") as f:
            json.dump(bf16_scores, f, ensure_ascii=False, indent=2)
    bf16_avg = resumed.get("bf16_avg") or (sum(bf16_scores.values()) / len(bf16_scores))
    print(f"[GQ] BF16 avg-{len(bf16_scores)}: {bf16_avg:.5f}")

    # ---------- 锚点: uniform-calib_bits (GPTQ) ----------
    if resumed:
        kl_cal = resumed["kl_cal"]
        anchor_scores = resumed.get("anchor_scores") or bf16_scores
        anchor_avg = resumed["anchor_recovery"] * bf16_avg
        anchor_rec = resumed["anchor_recovery"]
        print(f"[GQ] resume anchor: KL={kl_cal:.5f} recovery={anchor_rec:.5f} (跳过 GPTQ/eval)")
    else:
        print(f"[GQ] anchor: uniform-{calib_bits} (GPTQ), KL + benchmark...")
        apply_config_gptq(model, groups, [calib_bits] * len(groups), act, group_size, symmetric,
                          gptq_cache_dir=gptq_cache_dir)
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
    if args.sens_json:
        with open(args.sens_json, "r", encoding="utf-8") as f:
            raw_sens = json.load(f)
        # gq_sens.json 的 c_ear/c_kl 以 str key 存储, 转回 int
        sens = {"base_ear": raw_sens["base_ear"], "base_kl": raw_sens["base_kl"],
                "c_ear": {int(m): {int(b): v for b, v in cm.items()}
                          for m, cm in raw_sens["c_ear"].items()},
                "c_kl": {int(m): {int(b): v for b, v in cm.items()}
                         for m, cm in raw_sens["c_kl"].items()},
                "params": raw_sens["params"],
                "n_perm": raw_sens.get("n_perm", n_perm)}
        print(f"[GQ] load shapley sensitivity from {args.sens_json} "
              f"(base_kl={sens['base_kl']:.5f})")
    else:
        print(f"[GQ] shapley sensitivity (P={n_perm}, sens_n={sens_n}, bits={bits_list})...")
        sens = shapley_sensitivity(model, groups, act, metrics_fn, bits_list, n_perm=n_perm,
                                   group_size=group_size, symmetric=symmetric,
                                   seed=cfg.get("seed", 2),
                                   gptq_cache_dir=gptq_cache_dir)
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

    # ---------- TL 单点标定 + 预测二分 + re-anchor (论文 Algorithm 2) ----------
    from slq.gptq_shapley import predicted_kl
    alpha = (1.0 - anchor_rec) / max(kl_cal, 1e-6)
    d_thresh = (1.0 - target_rec) / alpha
    if args.thresh_scale != 1.0:
        d_thresh_orig = d_thresh
        d_thresh *= args.thresh_scale
        print(f"[GQ] thresh_scale={args.thresh_scale}: D_thresh {d_thresh_orig:.5f} -> {d_thresh:.5f}")
    d_pred_anchor = predicted_kl(sens, [calib_bits] * len(groups))
    rho = kl_cal / max(d_pred_anchor, 1e-8)
    print(f"[GQ] single-point: alpha={alpha:.4f} D_thresh={d_thresh:.5f} "
          f"D_pred_anchor={d_pred_anchor:.5f} rho={rho:.4f}")

    # 论文 Algorithm 2 的 re-anchor 循环:
    #   guardrail 失败 (实测/预测 相对锚点 rho 偏离 >2x) 时, 用实测 KL 重新估计 rho,
    #   并把搜索下界抬升到失败位宽, 重新二分, 直到 guardrail 通过或达到最大迭代。
    from slq.gptq_shapley import search_tl_predicted
    max_reanchor = cfg.get("max_reanchor", 4)
    failed_bits = 0.0
    final_alloc = None
    best_kl_actual = None
    for re_i in range(max_reanchor + 1):
        alloc, avg_bits = search_tl_predicted(sens, bits_list, d_thresh, rho,
                                              skip_below=failed_bits)
        if alloc is None:
            raise RuntimeError("TL search failed (no feasible config)")
        desc = describe_config(groups, alloc)
        print(f"[GQ][iter{re_i}] predicted alloc: avg_bits={desc['avg_bits']} "
              f"per_bits={desc['per_bits']}")

        restore_weights(groups, snapshot)
        apply_config_gptq(model, groups, alloc, act, group_size, symmetric,
                          gptq_cache_dir=gptq_cache_dir)
        # guardrail 的 KL 必须与预测/锚点在【同一批样本】上测, 否则样本数差异本身
        # 就会污染 ratio (论文单点标定协议: 锚点与候选同协议)。sens 集用于校验,
        # 全量集只用于最终 lm_eval benchmark。
        _, kl_actual = model_metrics(model, sens_inputs, sens_ref, device, topk=topk,
                                     temperature=temperature)
        d_pred = predicted_kl(sens, alloc)
        rho_actual = kl_actual / max(d_pred, 1e-8)
        ratio = rho_actual / max(rho, 1e-8)
        print(f"[GQ][iter{re_i}] guardrail(sens): actual_kl={kl_actual:.5f} "
              f"predicted_kl={d_pred:.5f} rho_actual={rho_actual:.4f} "
              f"ratio_vs_anchor_rho={ratio:.3f}")

        # guardrail 硬判据: 实测 KL 必须落在 (缩放后的) 阈值内 —— 线性恢复模型
        # 在低位宽会低估实际任务损失, 单靠 ratio 检查会放行超阈值的配置 (见首轮
        # recovery=0.9854 < 0.99)。ratio 仍作为 rho 偏离的软诊断。
        kl_ok = kl_actual <= d_thresh
        if 0.5 <= ratio <= 2.0 and kl_ok:
            final_alloc = alloc
            best_kl_actual = kl_actual
            result["guardrail"] = "ok"
            result["reanchor_iters"] = re_i
            print(f"[GQ] guardrail OK at iter {re_i} (kl_actual={kl_actual:.5f} <= d_thresh={d_thresh:.5f})")
            break
        # re-anchor: 用实测 KL 重估 rho, 并强制后续搜索位宽 >= 当前实测位宽
        failed_bits = desc["avg_bits"]
        rho = rho_actual
        print(f"[GQ] guardrail violated at iter {re_i} (ratio={ratio:.3f}, "
              f"kl_ok={kl_ok}), re-anchor rho->{rho:.4f}, floor avg_bits->{failed_bits:.3f}")
        result.setdefault("reanchor_hist", []).append(
            {"iter": re_i, "avg_bits": desc["avg_bits"], "kl_actual": kl_actual,
             "predicted_kl": d_pred, "rho_actual": rho_actual, "ratio": ratio})
        best_kl_actual = kl_actual  # 未通过时记录最后一次实测, 便于人工判断
    else:
        # 跑满仍未通过: 用最后一次配置并标记
        final_alloc = alloc
        result["guardrail"] = "unverified"

    alloc = final_alloc
    desc = describe_config(groups, alloc)
    result["avg_bits"] = desc["avg_bits"]
    result["alloc"] = alloc
    result["per_bits"] = desc["per_bits"]
    result["alpha"] = alpha
    result["d_thresh"] = d_thresh
    result["rho"] = rho
    result["d_pred_anchor"] = d_pred_anchor
    result["predicted_kl"] = predicted_kl(sens, alloc)
    result["kl_actual"] = best_kl_actual

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
