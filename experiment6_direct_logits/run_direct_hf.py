import argparse
import csv
import hashlib
import json
import math
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

# ============================================================
# EXPERIMENT 6 — DIRECT FULL-LOGITS REPLICATION
#
# Key difference from the Ollama runner:
#   - no HTTP request per token
#   - one full causal forward pass exposes next-token logits
#     for every reference position at once
#   - full-vocabulary entropy is available directly
#
# This is a separate replication backend, not a continuation of
# the quantized Ollama confirmatory run.
# ============================================================

HERE = Path(__file__).resolve().parent
TASKS_FILE = HERE / "tasks.json"

DEFAULT_MODEL = "Qwen/Qwen3-4B"
DEFAULT_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
DEFAULT_RESULTS_DIR = HERE / "results_direct_hf"
DEFAULT_BATCH_SIZE = 8
ORDER_SEED = 20260916
TOP_K = 20

# We intentionally preserve the exact plain raw context used by
# the Ollama teacher-forced experiment. No chat template and no
# representation instruction are added.
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
    completed = set()
    with trajectories_file.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            completed.add(row["trajectory_id"])
    return completed


def repair_partial_token_rows(tokens_file, completed):
    """
    Keep only token rows whose trajectory has a completed trajectory summary.
    This makes resume safe after interruption during CSV writing.
    """
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
            for variant_index, reference in enumerate(task[representation], start=1):
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
    Tokenize base+reference as ONE string.

    This deliberately fixes a subtle boundary issue that can occur when the
    reference is tokenized separately from the preceding base prompt.

    We require tokenization(base + reference) to begin with tokenization(base).
    With the frozen current prompts this should hold; if it does not, we stop
    rather than silently measure misaligned token positions.
    """
    base_text = build_base_prompt(spec["task_text"])
    full_text = base_text + spec["reference"]

    base_ids = tokenizer.encode(base_text, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    if full_ids[:len(base_ids)] != base_ids:
        raise RuntimeError(
            "Tokenizer boundary mismatch: full text does not preserve base token prefix.\n"
            f"trajectory={spec['trajectory_id']}\n"
            "This needs offset-based alignment before continuing."
        )

    reference_ids = full_ids[len(base_ids):]
    if not reference_ids:
        raise RuntimeError(f"Empty reference tokenization: {spec['trajectory_id']}")

    return {
        **spec,
        "base_text": base_text,
        "full_text": full_text,
        "base_ids": base_ids,
        "full_ids": full_ids,
        "reference_ids": reference_ids,
        "reference_start": len(base_ids),
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
    # Same primary exclusion as the Ollama protocol: exclude first reference token.
    valid = [r for r in rows if r["reference_position"] > 1]

    h20 = [r["entropy_top20_normalized"] for r in valid]
    hfull = [r["entropy_full"] for r in valid]
    p1 = [r["p1"] for r in valid]
    mass = [r["top20_mass"] for r in valid]
    high90 = [p >= 0.90 for p in p1]
    high99 = [p >= 0.99 for p in p1]
    deltas20 = [abs(h20[i] - h20[i - 1]) for i in range(1, len(h20))]
    deltasfull = [abs(hfull[i] - hfull[i - 1]) for i in range(1, len(hfull))]

    return {
        "token_count": len(rows),
        "analysis_token_count": len(valid),
        "special_token_events": 0,
        "special_token_fraction": 0.0,
        "trajectory_valid": True,
        # Comparable with Ollama confirmatory primary metric:
        "mean_entropy_top20": statistics.mean(h20),
        "median_entropy_top20": statistics.median(h20),
        "std_entropy_top20": statistics.pstdev(h20) if len(h20) > 1 else 0.0,
        "cumulative_entropy_top20": sum(h20),
        # Direct-backend advantage: exact full-vocabulary entropy.
        "mean_entropy_full": statistics.mean(hfull),
        "median_entropy_full": statistics.median(hfull),
        "std_entropy_full": statistics.pstdev(hfull) if len(hfull) > 1 else 0.0,
        "cumulative_entropy_full": sum(hfull),
        "mean_p1": statistics.mean(p1),
        "median_p1": statistics.median(p1),
        "mean_top20_mass": statistics.mean(mass),
        "fraction_p1_ge_090": sum(high90) / len(high90),
        "fraction_p1_ge_099": sum(high99) / len(high99),
        "longest_run_p1_ge_090": longest_true_run(high90),
        "longest_run_p1_ge_099": longest_true_run(high99),
        "mean_abs_entropy_change_top20": statistics.mean(deltas20) if deltas20 else 0.0,
        "mean_abs_entropy_change_full": statistics.mean(deltasfull) if deltasfull else 0.0,
        "reference_in_top20_fraction": (
            sum(bool(r["reference_in_top20"]) for r in valid) / len(valid)
        ),
        "mean_reference_surprisal": statistics.mean(
            r["reference_surprisal"] for r in valid
        ),
    }


def analyze_batch(prepared_batch, model, tokenizer, device, top_k=TOP_K):
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise RuntimeError("Tokenizer has no pad/eos token.")

    max_len = max(len(x["full_ids"]) for x in prepared_batch)
    batch_size = len(prepared_batch)

    input_ids = torch.full(
        (batch_size, max_len),
        fill_value=pad_id,
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

    batch_results = []

    for b, item in enumerate(prepared_batch):
        start = item["reference_start"]
        ref_ids = item["reference_ids"]

        # To predict token at absolute sequence position p, use logits[p - 1].
        prediction_positions = torch.arange(
            start - 1,
            start - 1 + len(ref_ids),
            device=device,
        )

        pred_logits = all_logits[b, prediction_positions, :].float()
        log_probs = torch.log_softmax(pred_logits, dim=-1)
        probs = torch.exp(log_probs)

        entropy_full = -(probs * log_probs).sum(dim=-1)

        k = min(top_k, pred_logits.shape[-1])
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

        # Exact rank is useful and still cheap for these short sequences.
        ref_logits = pred_logits.gather(1, ref_tensor.unsqueeze(1)).squeeze(1)
        ref_rank = (pred_logits > ref_logits.unsqueeze(1)).sum(dim=1) + 1

        # Bring only measurement arrays to CPU.
        entropy_full_cpu = entropy_full.cpu().tolist()
        entropy_top20_cpu = entropy_top20.cpu().tolist()
        p1_cpu = p1.cpu().tolist()
        p2_cpu = p2.cpu().tolist()
        top_mass_cpu = top_mass.cpu().tolist()
        ref_logprob_cpu = ref_logprob.cpu().tolist()
        ref_prob_cpu = ref_prob.cpu().tolist()
        ref_rank_cpu = ref_rank.cpu().tolist()
        top_ids_cpu = top_ids.cpu().tolist()
        top_log_probs_cpu = top_log_probs.cpu().tolist()
        top_probs_cpu = top_probs.cpu().tolist()

        rows = []

        for j, ref_id in enumerate(ref_ids):
            ids_j = top_ids_cpu[j]
            probs_j = top_probs_cpu[j]
            logps_j = top_log_probs_cpu[j]

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

            rank = int(ref_rank_cpu[j])
            row = {
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
                "entropy_full": float(entropy_full_cpu[j]),
                "entropy_top20_normalized": float(entropy_top20_cpu[j]),
                "p1": float(p1_cpu[j]),
                "p2": float(p2_cpu[j]),
                "p1_minus_p2": float(p1_cpu[j] - p2_cpu[j]),
                "top20_mass": float(top_mass_cpu[j]),
                "tail_mass": float(max(0.0, 1.0 - top_mass_cpu[j])),
                "reference_probability": float(ref_prob_cpu[j]),
                "reference_logprob": float(ref_logprob_cpu[j]),
                "reference_surprisal": float(-ref_logprob_cpu[j]),
                "reference_rank": rank,
                "reference_in_top20": rank <= top_k,
                "top1_token_id": int(ids_j[0]),
                "top1_token": top20[0]["token"],
                "top20_json": json.dumps(top20, ensure_ascii=False),
            }
            rows.append(row)

        batch_results.append((item, rows))

    # Free the very large vocabulary logits before next batch.
    del outputs, all_logits
    return batch_results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--revision", default=DEFAULT_REVISION)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--limit", type=int, default=0,
                   help="0 = all trajectories; otherwise run only first N specs.")
    p.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    p.add_argument("--local-files-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in PyTorch. Install a CUDA PyTorch wheel first."
        )

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)

    tasks = json.loads(TASKS_FILE.read_text(encoding="utf-8"))
    tasks_sha256 = hashlib.sha256(TASKS_FILE.read_bytes()).hexdigest()
    specs = build_specs(tasks)
    if args.limit > 0:
        specs = specs[:args.limit]

    results_dir = Path(args.results_dir).resolve()
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
    )

    print("Loading model in BF16 directly on GPU...")
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=args.local_files_only,
        low_cpu_mem_usage=True,
    )
    model.eval()
    load_seconds = time.perf_counter() - load_started

    prepared = [prepare_sequence(s, tokenizer) for s in specs]

    model_vocab = int(model.config.vocab_size)
    tokenizer_size = len(tokenizer)
    resolved_commit = getattr(model.config, "_commit_hash", None)
    if resolved_commit and resolved_commit != args.revision:
        raise RuntimeError(
            f"Resolved commit mismatch: {resolved_commit} != {args.revision}"
        )

    config = {
        "timestamp_started": datetime.now().isoformat(),
        "experiment": "Experiment 6 Direct Full-Logits Replication",
        "backend": "transformers_direct_forward",
        "model": args.model,
        "revision_requested": args.revision,
        "model_commit_hash": resolved_commit or args.revision,
        "model_dtype": "bfloat16",
        "tasks_sha256": tasks_sha256,
        "task_file": str(TASKS_FILE),
        "order_seed": ORDER_SEED,
        "top_k": TOP_K,
        "batch_size": args.batch_size,
        "trajectory_limit": args.limit,
        "trajectory_count": len(specs),
        "tokenizer_size": tokenizer_size,
        "model_vocab_size": model_vocab,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_runtime_reported_by_torch": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "visible_cuda_device_name": props.name,
        "visible_cuda_total_memory_gb": props.total_memory / (1024 ** 3),
        "model_load_seconds": load_seconds,
        "primary_comparable_metric": "mean_entropy_top20",
        "additional_exact_metric": "mean_entropy_full",
        "design": (
            "Same frozen tasks and references as confirmatory Ollama run. "
            "Raw Task/Solution context. Full base+reference string tokenized as "
            "one sequence. One causal forward pass exposes next-token logits "
            "for every teacher-forced reference position."
        ),
        "scientific_scope": (
            "Separate BF16 backend/precision replication; do not merge rows "
            "with the quantized Ollama confirmatory dataset."
        ),
    }

    if config_file.exists():
        old = json.loads(config_file.read_text(encoding="utf-8"))
        for key in (
            "tasks_sha256",
            "model",
            "revision_requested",
            "model_dtype",
            "top_k",
        ):
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

    print("=" * 110)
    print("EXPERIMENT 6 — DIRECT FULL-LOGITS REPLICATION")
    print("=" * 110)
    print(f"Model:                {args.model}")
    print(f"Backend:              Transformers direct forward")
    print(f"Precision:            BF16")
    print(f"GPU visible to run:   {props.name}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"Torch CUDA:           {torch.version.cuda}")
    print(f"Model load:           {format_duration(load_seconds)}")
    print(f"Batch size:           {args.batch_size}")
    print(f"Trajectories:         {len(prepared)}")
    print(f"Reference tokens:     {total_ref_tokens}")
    print(f"Already complete:     {len(completed)}")
    print(f"Remaining:            {len(remaining)}")
    print("=" * 110)

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
        batch_results = analyze_batch(batch, model, tokenizer, device)
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
                "base_token_count": len(item["base_ids"]),
                **summary,
            }

            append_rows(tokens_file, rows)
            append_rows(trajectories_file, [trajectory_row])

            done_this_session += 1

            print(
                f"  [{len(completed)+done_this_session:03d}/{len(prepared)}] "
                f"{item['trajectory_id']} | "
                f"H20={summary['mean_entropy_top20']:.4f} | "
                f"Hfull={summary['mean_entropy_full']:.4f} | "
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

    print("=" * 110)
    print("DIRECT RUN COMPLETE")
    print(f"Tokens:       {tokens_file}")
    print(f"Trajectories: {trajectories_file}")
    print(f"Config:       {config_file}")
    print("=" * 110)


if __name__ == "__main__":
    main()
