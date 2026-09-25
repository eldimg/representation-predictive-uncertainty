import argparse
import csv
import hashlib
import json
import os
import random
import statistics
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
TASKS_FILE = HERE / "tasks.json"
EXPECTED_TASKS_SHA256 = "dcdbc196f7dbf394fbf043690c764c470dbbbf9ee333e344c1f62f320a72a164"
ORDER_SEED = 20260916
TOP_K = 20
DEFAULT_BATCH_SIZE = 8


def build_base_prompt(task_text):
    return f"Task:\n{task_text}\n\nSolution:\n"


def format_duration(seconds):
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def safe_model_slug(model_name):
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in model_name)


def append_rows(path, rows):
    if not rows:
        return
    exists = path.exists() and path.stat().st_size > 0
    fieldnames = list(rows[0].keys())
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
        f.flush()


def load_completed(trajectories_file):
    if not trajectories_file.exists():
        return set()
    with trajectories_file.open("r", newline="", encoding="utf-8-sig") as f:
        return {row["trajectory_id"] for row in csv.DictReader(f)}


def repair_partial_token_rows(tokens_file, completed):
    if not tokens_file.exists() or tokens_file.stat().st_size == 0:
        return
    with tokens_file.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames:
        return
    kept = [r for r in rows if r.get("trajectory_id") in completed]
    removed = len(rows) - len(kept)
    if removed:
        print(f"Repairing token checkpoint: removing {removed} partial rows.")
        with tokens_file.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept)


def build_specs(tasks):
    specs = []
    for task in tasks:
        for representation in ("nl", "python"):
            refs = task[representation]
            if len(refs) != 3:
                raise RuntimeError(
                    f"{task['id']} / {representation}: expected 3 frozen references, found {len(refs)}"
                )
            for variant_index, reference in enumerate(refs, start=1):
                specs.append({
                    "trajectory_id": f"{task['id']}__{representation}__v{variant_index}",
                    "task_id": task["id"],
                    "family": task["family"],
                    "task_text": task["task"],
                    "representation": representation,
                    "variant": variant_index,
                    "reference": reference,
                })
    rng = random.Random(ORDER_SEED)
    rng.shuffle(specs)
    return specs


def prepare_sequence(spec, tokenizer):
    """
    Tokenize base+reference as one string.

    Fast tokenizers use character offsets to locate the first token touching
    the reference span. This handles BPE boundary fusion across the prompt /
    reference boundary. The primary metric excludes the first reference token.
    """
    base_text = build_base_prompt(spec["task_text"])
    full_text = base_text + spec["reference"]
    ref_char_start = len(base_text)

    boundary_crossed = False
    if getattr(tokenizer, "is_fast", False):
        enc = tokenizer(
            full_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        full_ids = list(enc["input_ids"])
        offsets = list(enc["offset_mapping"])
        reference_start = None
        for i, (start, end) in enumerate(offsets):
            if end > ref_char_start:
                reference_start = i
                boundary_crossed = start < ref_char_start
                break
        if reference_start is None:
            raise RuntimeError(f"Could not locate reference span: {spec['trajectory_id']}")
        alignment_method = "fast_offsets"
        base_token_count_metadata = len(
            tokenizer.encode(base_text, add_special_tokens=False)
        )
    else:
        base_ids = tokenizer.encode(base_text, add_special_tokens=False)
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        if full_ids[:len(base_ids)] != base_ids:
            raise RuntimeError(
                "Slow tokenizer changed the boundary between base prompt and reference. "
                f"trajectory={spec['trajectory_id']}. Use a fast tokenizer."
            )
        reference_start = len(base_ids)
        alignment_method = "prefix_exact"
        base_token_count_metadata = len(base_ids)

    reference_ids = full_ids[reference_start:]
    if not reference_ids:
        raise RuntimeError(f"Empty reference tokenization: {spec['trajectory_id']}")
    if reference_start < 1:
        raise RuntimeError(f"Reference starts at token 0: {spec['trajectory_id']}")

    return {
        **spec,
        "full_ids": full_ids,
        "reference_ids": reference_ids,
        "reference_start": reference_start,
        "base_token_count_metadata": base_token_count_metadata,
        "alignment_method": alignment_method,
        "boundary_crossed": boundary_crossed,
    }


def longest_true_run(values):
    best = 0
    current = 0
    for value in values:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def summarize(rows):
    valid = [r for r in rows if r["reference_position"] > 1]
    if not valid:
        raise RuntimeError("No analyzable tokens after first-token exclusion.")
    h20 = [r["entropy_top20_normalized"] for r in valid]
    hv = [r["entropy_valid_vocab"] for r in valid]
    p1 = [r["p1"] for r in valid]
    mass = [r["top20_mass"] for r in valid]
    high90 = [p >= 0.90 for p in p1]
    high99 = [p >= 0.99 for p in p1]
    d20 = [abs(h20[i] - h20[i - 1]) for i in range(1, len(h20))]
    dv = [abs(hv[i] - hv[i - 1]) for i in range(1, len(hv))]
    return {
        "token_count": len(rows),
        "analysis_token_count": len(valid),
        "trajectory_valid": True,
        "mean_entropy_top20": statistics.mean(h20),
        "median_entropy_top20": statistics.median(h20),
        "std_entropy_top20": statistics.pstdev(h20) if len(h20) > 1 else 0.0,
        "cumulative_entropy_top20": sum(h20),
        "mean_entropy_valid_vocab": statistics.mean(hv),
        "median_entropy_valid_vocab": statistics.median(hv),
        "std_entropy_valid_vocab": statistics.pstdev(hv) if len(hv) > 1 else 0.0,
        "cumulative_entropy_valid_vocab": sum(hv),
        "mean_p1": statistics.mean(p1),
        "median_p1": statistics.median(p1),
        "mean_top20_mass": statistics.mean(mass),
        "fraction_p1_ge_090": sum(high90) / len(high90),
        "fraction_p1_ge_099": sum(high99) / len(high99),
        "longest_run_p1_ge_090": longest_true_run(high90),
        "longest_run_p1_ge_099": longest_true_run(high99),
        "mean_abs_entropy_change_top20": statistics.mean(d20) if d20 else 0.0,
        "mean_abs_entropy_change_valid_vocab": statistics.mean(dv) if dv else 0.0,
        "reference_in_top20_fraction": (
            sum(bool(r["reference_in_top20"]) for r in valid) / len(valid)
        ),
        "mean_reference_surprisal": statistics.mean(
            r["reference_surprisal"] for r in valid
        ),
    }


def analyze_batch(prepared_batch, model, tokenizer, device, valid_vocab_size, top_k=TOP_K):
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0

    max_len = max(len(x["full_ids"]) for x in prepared_batch)
    batch_size = len(prepared_batch)

    input_ids = torch.full(
        (batch_size, max_len),
        fill_value=int(pad_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (batch_size, max_len),
        dtype=torch.long,
        device=device,
    )

    for b, item in enumerate(prepared_batch):
        ids = torch.tensor(item["full_ids"], dtype=torch.long, device=device)
        if int(ids.max()) >= valid_vocab_size:
            raise RuntimeError(
                f"Tokenizer produced id {int(ids.max())}, usable vocab is {valid_vocab_size}."
            )
        input_ids[b, :len(ids)] = ids
        attention_mask[b, :len(ids)] = 1

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )

    all_logits = outputs.logits
    if all_logits.shape[-1] < valid_vocab_size:
        raise RuntimeError(
            f"Model logits vocab {all_logits.shape[-1]} < valid vocab {valid_vocab_size}"
        )

    batch_results = []

    for b, item in enumerate(prepared_batch):
        start = item["reference_start"]
        ref_ids = item["reference_ids"]
        prediction_positions = torch.arange(
            start - 1,
            start - 1 + len(ref_ids),
            device=device,
        )

        # Exclude padded/unused LM-head rows beyond tokenizer-addressable ids.
        pred_logits = all_logits[b, prediction_positions, :valid_vocab_size].float()
        log_probs = torch.log_softmax(pred_logits, dim=-1)
        probs = torch.exp(log_probs)

        entropy_valid = -(probs * log_probs).sum(dim=-1)

        k = min(top_k, valid_vocab_size)
        top_log_probs, top_ids = torch.topk(log_probs, k=k, dim=-1)
        top_probs = torch.exp(top_log_probs)
        top_mass = top_probs.sum(dim=-1)
        q = top_probs / top_mass.unsqueeze(-1)
        entropy_top20 = -(q * torch.log(q)).sum(dim=-1)

        p1 = top_probs[:, 0]
        p2 = top_probs[:, 1] if k >= 2 else torch.zeros_like(p1)

        ref_tensor = torch.tensor(ref_ids, dtype=torch.long, device=device)
        ref_logprob = log_probs.gather(1, ref_tensor.unsqueeze(1)).squeeze(1)
        ref_prob = torch.exp(ref_logprob)
        ref_logits = pred_logits.gather(1, ref_tensor.unsqueeze(1)).squeeze(1)
        ref_rank = (pred_logits > ref_logits.unsqueeze(1)).sum(dim=1) + 1

        arrays = {
            "entropy_valid": entropy_valid.cpu().tolist(),
            "entropy_top20": entropy_top20.cpu().tolist(),
            "p1": p1.cpu().tolist(),
            "p2": p2.cpu().tolist(),
            "top_mass": top_mass.cpu().tolist(),
            "ref_logprob": ref_logprob.cpu().tolist(),
            "ref_prob": ref_prob.cpu().tolist(),
            "ref_rank": ref_rank.cpu().tolist(),
            "top_ids": top_ids.cpu().tolist(),
            "top_log_probs": top_log_probs.cpu().tolist(),
            "top_probs": top_probs.cpu().tolist(),
        }

        rows = []
        for j, ref_id in enumerate(ref_ids):
            ids_j = arrays["top_ids"][j]
            probs_j = arrays["top_probs"][j]
            logps_j = arrays["top_log_probs"][j]
            top20 = []
            for tok_id, prob, logp in zip(ids_j, probs_j, logps_j):
                top20.append({
                    "token_id": int(tok_id),
                    "token": tokenizer.decode(
                        [int(tok_id)],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    ),
                    "probability": float(prob),
                    "logprob": float(logp),
                })

            rank = int(arrays["ref_rank"][j])
            rows.append({
                "trajectory_id": item["trajectory_id"],
                "task_id": item["task_id"],
                "family": item["family"],
                "representation": item["representation"],
                "variant": item["variant"],
                "reference_position": j + 1,
                "absolute_token_position": start + j,
                "normalized_position": (
                    j / (len(ref_ids) - 1) if len(ref_ids) > 1 else 0.0
                ),
                "reference_token_id": int(ref_id),
                "reference_token": tokenizer.decode(
                    [int(ref_id)],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                "entropy_valid_vocab": float(arrays["entropy_valid"][j]),
                "entropy_top20_normalized": float(arrays["entropy_top20"][j]),
                "p1": float(arrays["p1"][j]),
                "p2": float(arrays["p2"][j]),
                "p1_minus_p2": float(arrays["p1"][j] - arrays["p2"][j]),
                "top20_mass": float(arrays["top_mass"][j]),
                "tail_mass": float(max(0.0, 1.0 - arrays["top_mass"][j])),
                "reference_probability": float(arrays["ref_prob"][j]),
                "reference_logprob": float(arrays["ref_logprob"][j]),
                "reference_surprisal": float(-arrays["ref_logprob"][j]),
                "reference_rank": rank,
                "reference_in_top20": rank <= top_k,
                "top1_token_id": int(ids_j[0]),
                "top1_token": top20[0]["token"],
                "top20_json": json.dumps(top20, ensure_ascii=False),
            })

        batch_results.append((item, rows))

    del outputs, all_logits
    return batch_results


def parse_dtype(value):
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    key = value.lower()
    if key not in mapping:
        raise argparse.ArgumentTypeError(f"Unsupported dtype: {value}")
    return mapping[key]


def dtype_name(dtype):
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.float32:
        return "float32"
    return str(dtype)


def parse_args():
    p = argparse.ArgumentParser(
        description="Experiment 7: frozen cross-model replication."
    )
    p.add_argument("--model", required=True, help="Hugging Face causal LM id or local path.")
    p.add_argument("--revision", default="main")
    p.add_argument("--dtype", type=parse_dtype, default=torch.bfloat16)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--limit", type=int, default=0,
                   help="0 = all 600; otherwise smoke-test first N.")
    p.add_argument("--results-dir", default=None)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in PyTorch.")
    if args.batch_size < 1:
        raise RuntimeError("--batch-size must be >= 1")

    actual_hash = hashlib.sha256(TASKS_FILE.read_bytes()).hexdigest()
    if actual_hash != EXPECTED_TASKS_SHA256:
        raise RuntimeError(
            "Frozen tasks.json hash changed.\n"
            f"expected={EXPECTED_TASKS_SHA256}\nactual={actual_hash}"
        )

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)

    tasks = json.loads(TASKS_FILE.read_text(encoding="utf-8"))
    if len(tasks) != 100:
        raise RuntimeError(f"Expected 100 tasks, found {len(tasks)}")

    specs = build_specs(tasks)
    if len(specs) != 600:
        raise RuntimeError(f"Expected 600 trajectories, found {len(specs)}")
    if args.limit > 0:
        specs = specs[:args.limit]

    if args.results_dir:
        results_dir = Path(args.results_dir).resolve()
    else:
        results_dir = (HERE / "results" / safe_model_slug(args.model)).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    tokens_file = results_dir / "tokens.csv"
    trajectories_file = results_dir / "trajectories.csv"
    config_file = results_dir / "config.json"
    order_file = results_dir / "trajectory_order.json"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        use_fast=True,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )

    print(f"Loading model in {dtype_name(args.dtype)} directly on GPU...")
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=args.dtype,
        device_map={"": 0},
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.eval()
    load_seconds = time.perf_counter() - load_started

    model_vocab = int(model.config.vocab_size)
    tokenizer_size = len(tokenizer)
    if tokenizer_size > model_vocab:
        raise RuntimeError(
            f"Tokenizer size ({tokenizer_size}) exceeds model vocab ({model_vocab})."
        )
    valid_vocab_size = tokenizer_size

    prepared = [prepare_sequence(s, tokenizer) for s in specs]
    boundary_crossings = sum(int(x["boundary_crossed"]) for x in prepared)

    config = {
        "timestamp_started": datetime.now().isoformat(),
        "experiment": "Experiment 7 Cross-Model Replication",
        "backend": "transformers_direct_forward",
        "model": args.model,
        "revision_requested": args.revision,
        "model_commit_hash": getattr(model.config, "_commit_hash", None),
        "model_type": getattr(model.config, "model_type", None),
        "architectures": getattr(model.config, "architectures", None),
        "model_dtype": dtype_name(args.dtype),
        "tasks_sha256": actual_hash,
        "order_seed": ORDER_SEED,
        "top_k": TOP_K,
        "batch_size": args.batch_size,
        "trajectory_limit": args.limit,
        "trajectory_count": len(specs),
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "tokenizer_size": tokenizer_size,
        "model_vocab_size": model_vocab,
        "valid_vocab_size_used": valid_vocab_size,
        "boundary_crossing_trajectories": boundary_crossings,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_runtime_reported_by_torch": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "visible_cuda_device_name": props.name,
        "visible_cuda_total_memory_gb": props.total_memory / (1024 ** 3),
        "model_load_seconds": load_seconds,
        "primary_metric": "mean_entropy_top20",
        "secondary_metric": "mean_entropy_valid_vocab",
        "primary_first_reference_token_excluded": True,
        "design": (
            "Frozen Experiment 6 stimuli and order. Same raw Task/Solution prompt. "
            "Full-sequence teacher forcing with direct causal logits. "
            "Within-model NL-vs-Python task-level comparison; no chat template."
        ),
        "cross_model_caution": (
            "Absolute entropy values across models are not directly comparable "
            "because tokenizers/vocabularies differ. Compare within-model deltas, "
            "sign consistency, confidence intervals, and standardized effect sizes."
        ),
    }

    if config_file.exists():
        old = json.loads(config_file.read_text(encoding="utf-8"))
        immutable = (
            "tasks_sha256", "model", "revision_requested", "model_dtype",
            "top_k", "valid_vocab_size_used", "trajectory_limit"
        )
        for key in immutable:
            if old.get(key) != config.get(key):
                raise RuntimeError(
                    f"Config changed for {key}: old={old.get(key)!r}, "
                    f"new={config.get(key)!r}. Refusing to resume."
                )
    else:
        config_file.write_text(
            json.dumps(config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    ids = [x["trajectory_id"] for x in prepared]
    if order_file.exists():
        saved_ids = json.loads(order_file.read_text(encoding="utf-8"))
        if saved_ids != ids:
            raise RuntimeError("Trajectory order changed. Refusing to resume.")
    else:
        order_file.write_text(json.dumps(ids, indent=2), encoding="utf-8")

    completed = load_completed(trajectories_file)
    repair_partial_token_rows(tokens_file, completed)
    remaining = [x for x in prepared if x["trajectory_id"] not in completed]

    total_ref_tokens = sum(len(x["reference_ids"]) for x in prepared)
    remaining_ref_tokens = sum(len(x["reference_ids"]) for x in remaining)

    print("=" * 112)
    print("EXPERIMENT 7 — CROSS-MODEL DIRECT-LOGITS REPLICATION")
    print("=" * 112)
    print(f"Model:                 {args.model}")
    print(f"Revision:              {args.revision}")
    print(f"Model type:            {getattr(model.config, 'model_type', None)}")
    print(f"Precision:             {dtype_name(args.dtype)}")
    print(f"GPU visible to run:    {props.name}")
    print(f"CUDA_VISIBLE_DEVICES:  {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"Tokenizer size:        {tokenizer_size}")
    print(f"Model vocab size:      {model_vocab}")
    print(f"Valid vocab measured:  {valid_vocab_size}")
    print(f"Boundary crossings:    {boundary_crossings}")
    print(f"Model load:            {format_duration(load_seconds)}")
    print(f"Batch size:            {args.batch_size}")
    print(f"Trajectories:          {len(prepared)}")
    print(f"Reference tokens:      {total_ref_tokens}")
    print(f"Already complete:      {len(completed)}")
    print(f"Remaining:             {len(remaining)}")
    print("=" * 112)

    if not remaining:
        print("Nothing to do.")
        return

    session_started = time.perf_counter()
    recent_batch_rates = deque(maxlen=10)
    done_this_session = 0

    for batch_start in range(0, len(remaining), args.batch_size):
        batch = remaining[batch_start:batch_start + args.batch_size]
        batch_token_count = sum(len(x["reference_ids"]) for x in batch)

        started = time.perf_counter()
        batch_results = analyze_batch(
            batch, model, tokenizer, device, valid_vocab_size
        )
        compute_seconds = time.perf_counter() - started

        for item, rows in batch_results:
            summary = summarize(rows)
            trajectory_row = {
                "trajectory_id": item["trajectory_id"],
                "task_id": item["task_id"],
                "family": item["family"],
                "representation": item["representation"],
                "variant": item["variant"],
                "task_text": item["task_text"],
                "reference_solution": item["reference"],
                "alignment_method": item["alignment_method"],
                "boundary_crossed": item["boundary_crossed"],
                "reference_start_token_index": item["reference_start"],
                "base_token_count_metadata": item["base_token_count_metadata"],
                **summary,
            }
            append_rows(tokens_file, rows)
            append_rows(trajectories_file, [trajectory_row])

            done_this_session += 1
            print(
                f"  [{len(completed)+done_this_session:03d}/{len(prepared)}] "
                f"{item['trajectory_id']} | "
                f"H20={summary['mean_entropy_top20']:.4f} | "
                f"Hvalid={summary['mean_entropy_valid_vocab']:.4f} | "
                f"p1={summary['mean_p1']:.4f}"
            )

        rate = batch_token_count / max(compute_seconds, 1e-9)
        recent_batch_rates.append(rate)
        remaining_ref_tokens -= batch_token_count

        avg_rate = sum(recent_batch_rates) / len(recent_batch_rates)
        eta = remaining_ref_tokens / max(avg_rate, 1e-9)
        elapsed = time.perf_counter() - session_started

        print(
            f"BATCH {batch_start//args.batch_size + 1} | "
            f"{len(batch)} trajectories | {batch_token_count} ref tokens | "
            f"compute {compute_seconds:.2f}s | "
            f"{rate:.1f} ref-tok/s | "
            f"elapsed {format_duration(elapsed)} | "
            f"ETA {format_duration(eta)}"
        )

    print("=" * 112)
    print("EXPERIMENT 7 MODEL RUN COMPLETE")
    print(f"Tokens:       {tokens_file}")
    print(f"Trajectories: {trajectories_file}")
    print(f"Config:       {config_file}")
    print("=" * 112)


if __name__ == "__main__":
    main()
