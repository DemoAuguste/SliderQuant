# -*- coding: utf-8 -*-
"""SLQ 入口: 逐层非均匀混合精度量化 (统计无损) 的 P0 复现。

用法:
    python main_slq.py --config configs/qwen3-0.6b-slq-base4.yaml --mode base
    python main_slq.py --config configs/qwen3-0.6b-slq-dl.yaml   --mode dl
    python main_slq.py --config configs/qwen3-0.6b-slq-tl.yaml  --mode tl \
        --anchor_recovery 0.96
    python main_slq.py --config ... --mode eval --apply_config <json>
"""
import argparse
import json
import os
import time

import torch

from slq.pipeline import (apply_config, apply_uniform, build_groups, describe_config,
                          get_device, load_model, run_dl_search, run_tl_search)
from slq.metrics import collect_reference, model_metrics


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", required=True,
                    choices=["base", "dl", "tl", "eval", "ref"])
    ap.add_argument("--output_dir", default="log/slq")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--apply_config", default=None, help="json 位宽配置 (eval mode)")
    ap.add_argument("--anchor_recovery", type=float, default=None, help="TL 锚点恢复率")
    ap.add_argument("--eval_tasks", default=None, help="逗号分隔任务, 覆盖 config")
    ap.add_argument("--skip_bench", action="store_true", help="TL 跳过 benchmark")
    return ap.parse_args()


def load_calibration(args, cfg, model_path):
    """加载校准数据。优先读缓存, 否则走 datautils.get_loaders。"""
    nsamples = cfg.get("nsamples", 128)
    seqlen = cfg.get("seqlen", 2048)
    calib = cfg.get("calib_dataset", "wikitext2")
    net = cfg.get("net", "qwen3")
    cache_dir = args.cache_dir or cfg.get("cache_dir", "cache")
    fn = f"dataloader_{net}_{calib}_{nsamples}"
    if seqlen != 2048:
        fn += f"_{seqlen}"
    fn += ".cache"
    cache_path = os.path.join(cache_dir, fn)
    if os.path.exists(cache_path):
        print(f"[calib] load cache {cache_path}")
        trainloader = torch.load(cache_path, weights_only=False, map_location="cpu")
        inputs = [inp.clone() for inp, _ in trainloader]
        return inputs

    print(f"[calib] cache {cache_path} not found, fallback to datautils.get_loaders")
    from datautils import get_loaders
    trainloader, _ = get_loaders(calib, nsamples=nsamples, seed=cfg.get("seed", 2),
                                 seqlen=seqlen, model=model_path)
    return [inp.clone() for inp, _ in trainloader]


def snapshot_weights(groups):
    return [m.weight.detach().clone() for grp in groups for m in grp["modules"]]


@torch.no_grad()
def restore_weights(groups, snapshot):
    i = 0
    for grp in groups:
        for mod in grp["modules"]:
            mod.weight.copy_(snapshot[i])
            i += 1


def eval_lm_eval(model, tokenizer, tasks, batch_size="8", device=None):
    """在进程内跑 lm_eval (HFLM + simple_evaluate)。返回 {task: acc}。"""
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager
    if device is None:
        device = "npu" if torch.npu.is_available() else "cpu"
    try:
        hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, device=device)
    except TypeError:
        hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
        hflm._device = torch.device(device)
    tm = TaskManager(include_defaults=True)
    res = lm_eval.simple_evaluate(hflm, tasks=tasks.split(","), batch_size=batch_size,
                                  task_manager=tm)["results"]
    return {t: res[t].get("acc_norm,none", res[t].get("acc,none")) for t in tasks.split(",")}


def eval_ppl(model, tokenizer, seqlen, test_inputs, device):
    """简单 PPL: 在给定 input_ids 上计算平均负对数似然。"""
    model.eval()
    nlls = 0.0
    cnt = 0
    with torch.no_grad():
        for inp in test_inputs:
            batch = inp.to(device)
            logits = model(batch).logits.float()
            shift_logits = logits[:, :-1, :]
            shift_labels = batch[:, 1:]
            loss = torch.nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                shift_labels.reshape(-1), reduction="sum")
            nlls += loss.item()
            cnt += shift_labels.numel()
    return float(torch.exp(torch.tensor(nlls / max(cnt, 1), dtype=torch.float32)))


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
    print(f"[SLQ] device={device} config={args.config} mode={args.mode}")

    model, tokenizer, device = load_model(cfg["model"], cfg.get("use_bfloat16", True), device)
    groups = build_groups(model, cfg.get("group_mode", "split"))
    print(f"[SLQ] groups: {len(groups)}")
    inputs = load_calibration(args, cfg, cfg["model"])
    topk = cfg.get("topk", 10)
    temperature = cfg.get("temperature", 1.0)
    ref = collect_reference(model, inputs, device, topk=topk, temperature=temperature)
    print("[SLQ] reference cache built")
    snapshot = snapshot_weights(groups)
    bits_list = cfg["bits_list"]

    result = {"config": cfg, "group_mode": cfg.get("group_mode"), "bits_list": bits_list,
              "n_groups": len(groups)}

    if args.mode == "base":
        # 参考自对比 (top-10 质量) 作为 EAR 上界
        ear_self, kl_self = model_metrics(model, inputs, ref, device, topk=topk,
                                          temperature=temperature)
        print(f"[base] original vs original: EAR={ear_self:.5f} KL={kl_self:.6f} (top-10 质量)")
        result["self_ear"] = ear_self
        result["self_kl"] = kl_self
        # uniform 4-bit (asym) 基线 + symmetric 对照 (gamma^2 实验)
        for bits, sym in [(4, False), (4, True)]:
            apply_uniform(model, groups, bits, cfg.get("group_size", 128), symmetric=sym)
            ear, kl = model_metrics(model, inputs, ref, device, topk=topk,
                                    temperature=temperature)
            print(f"[base] uniform-{bits} symmetric={sym}: EAR={ear:.5f} KL={kl:.5f}")
            restore_weights(groups, snapshot)
            result[f"uniform{bits}_{'sym' if sym else 'asym'}"] = {"ear": ear, "kl": kl}
        for b in [3, 5]:
            apply_uniform(model, groups, b, cfg.get("group_size", 128), symmetric=False)
            ear, kl = model_metrics(model, inputs, ref, device, topk=topk,
                                    temperature=temperature)
            print(f"[base] uniform-{b} asym: EAR={ear:.5f} KL={kl:.5f}")
            restore_weights(groups, snapshot)
            result[f"uniform{b}_asym"] = {"ear": ear, "kl": kl}

    elif args.mode == "dl":
        sens_data = None
        sens_cache_path = os.path.join(out_dir, "slq_dl_sens.json")
        if os.path.exists(sens_cache_path):
            print(f"[dl] load sensitivity cache {sens_cache_path}")
            with open(sens_cache_path, "r", encoding="utf-8") as f:
                sens_data = json.load(f)
        res = run_dl_search(model, groups, inputs, ref, device, cfg, sens_data)
        result.update(res)
        # 落盘敏感度数据库, 支持断点续跑
        with open(sens_cache_path, "w", encoding="utf-8") as f:
            json.dump(res["sens"], f, ensure_ascii=False, indent=2)
        restore_weights(groups, snapshot)
        result.pop("sens", None)

    elif args.mode == "tl":
        tasks = args.eval_tasks or cfg.get("eval_tasks", "piqa,arc_easy")
        calib_bits = cfg.get("tl_calib_bits", 6)
        sens_n = cfg.get("sens_nsamples", len(inputs))
        print(f"[tl] calibration: uniform-{calib_bits} benchmark + KL, tasks={tasks}")
        # BF16 基线 benchmark (带缓存)
        bf16_cache = os.path.join(out_dir, "bf16_scores.json")
        if os.path.exists(bf16_cache):
            with open(bf16_cache, "r", encoding="utf-8") as f:
                bf16_scores = json.load(f)
            print(f"[tl] load BF16 scores from {bf16_cache}")
        else:
            bf16_scores = eval_lm_eval(model, tokenizer, tasks, cfg.get("lm_eval_batch_size", "8"), device)
            with open(bf16_cache, "w", encoding="utf-8") as f:
                json.dump(bf16_scores, f, ensure_ascii=False, indent=2)
        bf16_avg = sum(bf16_scores.values()) / len(bf16_scores)
        # 标定配置: 同 sens 子集测 KL + 全量 benchmark (带缓存)
        apply_uniform(model, groups, calib_bits, cfg.get("group_size", 128))
        _, kl_cal = model_metrics(model, inputs[:sens_n], ref[:sens_n], device, topk=topk,
                                  temperature=temperature)  # (ear, kl)
        anchor_cache = os.path.join(out_dir, f"anchor_{calib_bits}b_scores.json")
        if os.path.exists(anchor_cache):
            with open(anchor_cache, "r", encoding="utf-8") as f:
                anchor_scores = json.load(f)
            print(f"[tl] load anchor scores from {anchor_cache}")
        else:
            anchor_scores = eval_lm_eval(model, tokenizer, tasks, cfg.get("lm_eval_batch_size", "8"), device)
            with open(anchor_cache, "w", encoding="utf-8") as f:
                json.dump(anchor_scores, f, ensure_ascii=False, indent=2)
        restore_weights(groups, snapshot)
        anchor_avg = sum(anchor_scores.values()) / len(anchor_scores)
        anchor_rec = anchor_avg / max(bf16_avg, 1e-9)
        print(f"[tl] BF16 avg={bf16_avg:.4f} anchor({calib_bits}b) avg={anchor_avg:.4f} "
              f"recovery={anchor_rec:.4f} KL_cal={kl_cal:.5f}")
        # 复用 DL 敏感度缓存 (含 alpha/beta)
        sens_data = None
        sens_cache_path = os.path.join(out_dir, "slq_dl_sens.json")
        if os.path.exists(sens_cache_path):
            print(f"[tl] load sensitivity cache {sens_cache_path}")
            with open(sens_cache_path, "r", encoding="utf-8") as f:
                sens_data = json.load(f)
        res = run_tl_search(model, groups, inputs, ref, device, cfg, anchor_rec, kl_cal,
                            bf16_scores, sens_data)
        result.update(res)
        result["bf16_scores"] = bf16_scores
        result["anchor_scores"] = anchor_scores
        restore_weights(groups, snapshot)
        result.pop("sens", None)

    elif args.mode == "eval":
        apply_cfg_path = args.apply_config
        if apply_cfg_path is None:
            cand = os.path.join(out_dir, "slq_config.json")
            if os.path.exists(cand):
                apply_cfg_path = cand
            else:
                raise SystemExit("--mode eval 需要 --apply_config <json>")
        with open(apply_cfg_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        alloc = saved["alloc"]
        apply_config(model, groups, alloc, cfg.get("group_size", 128), cfg.get("symmetric", False))
        ear, kl = model_metrics(model, inputs, ref, device, topk=topk, temperature=temperature)
        print(f"[eval] EAR={ear:.5f} KL={kl:.5f} avg_bits={describe_config(groups, alloc)['avg_bits']}")
        result.update({"ear": ear, "kl": kl, "alloc": alloc})
        tasks = args.eval_tasks or cfg.get("eval_tasks", "")
        if tasks:
            print(f"[eval] lm_eval tasks: {tasks}")
            scores = eval_lm_eval(model, tokenizer, tasks, cfg.get("lm_eval_batch_size", "8"), device)
            print(f"[eval] scores: {scores}")
            result["scores"] = scores
        restore_weights(groups, snapshot)

    result["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out = os.path.join(out_dir, f"slq_{args.mode}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[SLQ] result saved to {out}")


if __name__ == "__main__":
    main()
