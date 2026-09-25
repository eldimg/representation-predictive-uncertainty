from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (
    CONDITIONS,
    MODEL_REGISTRY_FILE,
    ORDER_FILE,
    RENDERED_FILE,
    TOP_K,
    VALIDATION_FILE,
    append_csv_rows,
    atomic_write_json,
    build_prompt,
    is_punctuation_only,
    is_whitespace_only,
    load_json,
    map_token_to_semantic_step,
    sha256_file,
)


def load_model_entry(slug: str) -> dict[str, Any]:
    registry = load_json(MODEL_REGISTRY_FILE)
    matches = [model for model in registry["models"] if model["slug"] == slug]
    if len(matches) != 1:
        raise ValueError(f"Unknown model slug: {slug}")
    return matches[0]


def prepare_sequence(record: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    base = build_prompt(record["task_text"])
    full_text = base + record["reference_text"]
    reference_char_start = len(base)
    encoding = tokenizer(
        full_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    full_ids = list(encoding["input_ids"])
    offsets = [tuple(pair) for pair in encoding["offset_mapping"]]
    reference_start = None
    boundary_crossed = False
    for index, (start, end) in enumerate(offsets):
        if end > reference_char_start:
            reference_start = index
            boundary_crossed = start < reference_char_start
            break
    if reference_start is None or reference_start < 1:
        raise RuntimeError(f"Could not align reference: {record['trajectory_id']}")
    reference_ids = full_ids[reference_start:]
    if len(reference_ids) < 2:
        raise RuntimeError(f"Reference too short: {record['trajectory_id']}")
    return {
        **record,
        "base_text": base,
        "full_text": full_text,
        "full_ids": full_ids,
        "offsets": offsets,
        "reference_start": reference_start,
        "reference_char_start": reference_char_start,
        "reference_ids": reference_ids,
        "boundary_crossed": boundary_crossed,
    }


def classify_step_mapping(item: dict[str, Any], absolute_index: int) -> tuple[str, str | None]:
    if item["condition"] == "nl_exp7_anchor":
        return "not_applicable", None
    start, end = item["offsets"][absolute_index]
    piece = item["full_text"][start:end]
    if is_whitespace_only(piece):
        return "whitespace_only", None
    if is_punctuation_only(piece):
        return "punctuation_only", None
    return map_token_to_semantic_step(
        max(0, start - item["reference_char_start"]),
        end - item["reference_char_start"],
        item["semantic_step_spans"],
    )


def analyze_batch(
    prepared: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    device: torch.device,
    valid_vocab_size: int,
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0
    max_length = max(len(item["full_ids"]) for item in prepared)
    input_ids = torch.full(
        (len(prepared), max_length), int(pad_id), dtype=torch.long, device=device
    )
    attention_mask = torch.zeros_like(input_ids)
    for batch_index, item in enumerate(prepared):
        ids = torch.tensor(item["full_ids"], dtype=torch.long, device=device)
        if int(ids.max()) >= valid_vocab_size:
            raise RuntimeError(f"Tokenizer ID exceeds usable vocabulary: {item['trajectory_id']}")
        input_ids[batch_index, : len(ids)] = ids
        attention_mask[batch_index, : len(ids)] = 1

    with torch.inference_mode():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).logits

    output: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for batch_index, item in enumerate(prepared):
        start = item["reference_start"]
        reference_ids = item["reference_ids"]
        prediction_positions = torch.arange(
            start - 1, start - 1 + len(reference_ids), device=device
        )
        prediction_logits = logits[
            batch_index, prediction_positions, :valid_vocab_size
        ].float()
        log_probs = torch.log_softmax(prediction_logits, dim=-1)
        probs = torch.exp(log_probs)
        entropy_valid = -(probs * log_probs).sum(dim=-1)
        top_log_probs, top_ids = torch.topk(
            log_probs, k=min(TOP_K, valid_vocab_size), dim=-1
        )
        top_probs = torch.exp(top_log_probs)
        top_mass = top_probs.sum(dim=-1)
        normalized = top_probs / top_mass.unsqueeze(-1)
        entropy_top20 = -(normalized * torch.log(normalized)).sum(dim=-1)
        p1 = top_probs[:, 0]
        p2 = top_probs[:, 1]
        reference_tensor = torch.tensor(reference_ids, dtype=torch.long, device=device)
        reference_logprob = log_probs.gather(1, reference_tensor[:, None]).squeeze(1)
        reference_logits = prediction_logits.gather(1, reference_tensor[:, None]).squeeze(1)
        reference_rank = (prediction_logits > reference_logits[:, None]).sum(dim=1) + 1

        arrays = {
            "entropy_valid": entropy_valid.cpu().tolist(),
            "entropy_top20": entropy_top20.cpu().tolist(),
            "p1": p1.cpu().tolist(),
            "p2": p2.cpu().tolist(),
            "top_mass": top_mass.cpu().tolist(),
            "reference_logprob": reference_logprob.cpu().tolist(),
            "reference_rank": reference_rank.cpu().tolist(),
            "top_ids": top_ids.cpu().tolist(),
            "top_probs": top_probs.cpu().tolist(),
        }
        rows = []
        for position, reference_id in enumerate(reference_ids):
            absolute_token_index = start + position
            offset_start, offset_end = item["offsets"][absolute_token_index]
            mapping_status, semantic_step_id = classify_step_mapping(
                item, absolute_token_index
            )
            top_tokens = [
                {
                    "token_id": int(token_id),
                    "token": tokenizer.decode(
                        [int(token_id)],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    ),
                    "probability": float(probability),
                }
                for token_id, probability in zip(
                    arrays["top_ids"][position], arrays["top_probs"][position]
                )
            ]
            rows.append(
                {
                    "trajectory_id": item["trajectory_id"],
                    "program_id": item["program_id"],
                    "task_id": item["task_id"],
                    "family": item["family"],
                    "variant": item["variant"],
                    "condition": item["condition"],
                    "reference_position": position + 1,
                    "absolute_token_position": absolute_token_index,
                    "reference_token_id": int(reference_id),
                    "reference_token": tokenizer.decode(
                        [int(reference_id)],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    ),
                    "reference_char_start": offset_start - item["reference_char_start"],
                    "reference_char_end": offset_end - item["reference_char_start"],
                    "entropy_top20_normalized": float(arrays["entropy_top20"][position]),
                    "entropy_valid_vocab": float(arrays["entropy_valid"][position]),
                    "p1": float(arrays["p1"][position]),
                    "p2": float(arrays["p2"][position]),
                    "top20_mass": float(arrays["top_mass"][position]),
                    "reference_logprob": float(arrays["reference_logprob"][position]),
                    "reference_surprisal": float(-arrays["reference_logprob"][position]),
                    "reference_rank": int(arrays["reference_rank"][position]),
                    "reference_in_top20": int(arrays["reference_rank"][position]) <= TOP_K,
                    "semantic_mapping_status": (
                        "excluded_first_token" if position == 0 else mapping_status
                    ),
                    "semantic_step_id": semantic_step_id if position > 0 else None,
                    "top20_json": json.dumps(top_tokens, ensure_ascii=False),
                }
            )
        output.append((item, rows))
    del logits
    return output


def summarize_rows(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    analyzed = [row for row in rows if int(row["reference_position"]) > 1]
    if not analyzed:
        raise RuntimeError("No tokens after first-token exclusion")
    h20 = [float(row["entropy_top20_normalized"]) for row in analyzed]
    valid = [float(row["entropy_valid_vocab"]) for row in analyzed]
    p1 = [float(row["p1"]) for row in analyzed]
    mass = [float(row["top20_mass"]) for row in analyzed]
    surprisal = [float(row["reference_surprisal"]) for row in analyzed]
    step_metric = None
    mapped_fraction = None
    boundary_fraction = None
    unmapped_fraction = None
    if condition != "nl_exp7_anchor":
        eligible = [
            row
            for row in analyzed
            if row["semantic_mapping_status"]
            not in {"whitespace_only", "punctuation_only", "not_applicable"}
        ]
        mapped = [row for row in eligible if row["semantic_mapping_status"] == "mapped"]
        by_step: dict[str, list[float]] = defaultdict(list)
        for row in mapped:
            by_step[str(row["semantic_step_id"])].append(
                float(row["entropy_top20_normalized"])
            )
        if by_step:
            step_metric = statistics.mean(
                statistics.mean(values) for values in by_step.values()
            )
        denominator = len(eligible)
        mapped_fraction = len(mapped) / denominator if denominator else 0.0
        boundary_fraction = (
            sum(row["semantic_mapping_status"] == "boundary_overlap" for row in eligible)
            / denominator
            if denominator
            else 0.0
        )
        unmapped_fraction = (
            sum(row["semantic_mapping_status"] == "unmapped" for row in eligible)
            / denominator
            if denominator
            else 0.0
        )
    return {
        "token_count": len(rows),
        "analysis_token_count": len(analyzed),
        "trajectory_valid": True,
        "mean_entropy_top20": statistics.mean(h20),
        "median_entropy_top20": statistics.median(h20),
        "mean_entropy_valid_vocab": statistics.mean(valid),
        "median_entropy_valid_vocab": statistics.median(valid),
        "semantic_step_balanced_entropy_top20": step_metric,
        "mean_p1": statistics.mean(p1),
        "median_p1": statistics.median(p1),
        "mean_top20_mass": statistics.mean(mass),
        "mean_reference_surprisal": statistics.mean(surprisal),
        "median_reference_rank": statistics.median(
            float(row["reference_rank"]) for row in analyzed
        ),
        "fraction_p1_ge_090": sum(value >= 0.90 for value in p1) / len(p1),
        "fraction_p1_ge_099": sum(value >= 0.99 for value in p1) / len(p1),
        "step_mapped_token_fraction": mapped_fraction,
        "boundary_overlap_token_fraction": boundary_fraction,
        "unmapped_token_fraction": unmapped_fraction,
    }


def load_completed(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["trajectory_id"] for row in csv.DictReader(handle)}


def repair_partial_tokens(tokens_path: Path, completed: set[str]) -> None:
    if not tokens_path.exists() or tokens_path.stat().st_size == 0:
        return
    with tokens_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = [row for row in reader if row["trajectory_id"] in completed]
    temporary = tokens_path.with_suffix(".repair.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, tokens_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-slug", required=True, choices=("qwen3", "phi3", "mistral"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--results-dir")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--confirm-real-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.confirm_real_run:
        raise SystemExit(
            "Refusing to run real Experiment 8 trajectories without --confirm-real-run"
        )
    validation = load_json(VALIDATION_FILE)
    if not validation.get("passed"):
        raise RuntimeError("Stimulus validation report does not pass")
    if validation["hashes"]["rendered_file_sha256"] != sha256_file(RENDERED_FILE):
        raise RuntimeError("Rendered stimuli changed after validation")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    entry = load_model_entry(args.model_slug)
    records = load_json(RENDERED_FILE)["records"]
    by_id = {record["trajectory_id"]: record for record in records}
    order = load_json(ORDER_FILE)
    if set(order) != set(by_id) or len(order) != 1800:
        raise RuntimeError("Frozen trajectory order is invalid")
    ordered_records = [by_id[trajectory_id] for trajectory_id in order]

    results_dir = (
        Path(args.results_dir).resolve()
        if args.results_dir
        else (RENDERED_FILE.parent.parent / "results" / args.model_slug).resolve()
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    tokens_path = results_dir / "tokens.csv"
    trajectories_path = results_dir / "trajectories.csv"
    config_path = results_dir / "config.json"
    result_order_path = results_dir / "trajectory_order.json"

    tokenizer = AutoTokenizer.from_pretrained(
        entry["model_id"],
        revision=entry["revision"],
        use_fast=True,
        local_files_only=args.local_files_only,
        trust_remote_code=False,
    )
    device = torch.device("cuda:0")
    model = AutoModelForCausalLM.from_pretrained(
        entry["model_id"],
        revision=entry["revision"],
        dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=args.local_files_only,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model_vocab = int(model.config.vocab_size)
    tokenizer_size = len(tokenizer)
    if tokenizer_size > model_vocab:
        raise RuntimeError("Tokenizer vocabulary exceeds model vocabulary")
    valid_vocab_size = tokenizer_size
    resolved_commit = getattr(model.config, "_commit_hash", None)
    if resolved_commit and resolved_commit != entry["revision"]:
        raise RuntimeError(
            f"Resolved commit mismatch: {resolved_commit} != {entry['revision']}"
        )

    prepared = [prepare_sequence(record, tokenizer) for record in ordered_records]
    config = {
        "experiment": "Experiment 8 Sequential Representational Interventions",
        "timestamp_started_utc": datetime.now(timezone.utc).isoformat(),
        "model_slug": args.model_slug,
        "model": entry["model_id"],
        "revision_requested": entry["revision"],
        "model_commit_hash": resolved_commit or entry["revision"],
        "dtype": "bfloat16",
        "batch_size": args.batch_size,
        "trajectory_count": len(prepared),
        "conditions": list(CONDITIONS),
        "top_k": TOP_K,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_size": tokenizer_size,
        "model_vocab_size": model_vocab,
        "valid_vocab_size_used": valid_vocab_size,
        "boundary_crossing_trajectories": sum(
            int(item["boundary_crossed"]) for item in prepared
        ),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tasks_sha256": validation["hashes"]["tasks_file_sha256"],
        "ir_sha256": validation["hashes"]["ir_file_sha256"],
        "stimuli_sha256": validation["hashes"]["rendered_file_sha256"],
        "validation_report_sha256": sha256_file(VALIDATION_FILE),
        "primary_metric": "mean_entropy_top20",
        "robustness_metrics": [
            "mean_entropy_valid_vocab",
            "semantic_step_balanced_entropy_top20",
        ],
        "first_reference_token_excluded": True,
    }
    if config_path.exists():
        old = load_json(config_path)
        immutable = (
            "model",
            "revision_requested",
            "trajectory_count",
            "top_k",
            "tasks_sha256",
            "ir_sha256",
            "stimuli_sha256",
        )
        for key in immutable:
            if old.get(key) != config.get(key):
                raise RuntimeError(f"Resume config mismatch for {key}")
    else:
        atomic_write_json(config_path, config)
    if result_order_path.exists():
        if load_json(result_order_path) != order:
            raise RuntimeError("Resume trajectory order mismatch")
    else:
        atomic_write_json(result_order_path, order)

    completed = load_completed(trajectories_path)
    repair_partial_tokens(tokens_path, completed)
    remaining = [item for item in prepared if item["trajectory_id"] not in completed]
    print(
        f"Model={entry['model_id']} trajectories={len(prepared)} "
        f"completed={len(completed)} remaining={len(remaining)}"
    )
    rates: deque[float] = deque(maxlen=10)
    started = time.perf_counter()
    for offset in range(0, len(remaining), args.batch_size):
        batch = remaining[offset : offset + args.batch_size]
        batch_started = time.perf_counter()
        results = analyze_batch(batch, model, tokenizer, device, valid_vocab_size)
        elapsed = time.perf_counter() - batch_started
        token_count = sum(len(item["reference_ids"]) for item in batch)
        rates.append(token_count / max(elapsed, 1e-9))
        for item, token_rows in results:
            summary = summarize_rows(token_rows, item["condition"])
            trajectory_row = {
                "trajectory_id": item["trajectory_id"],
                "program_id": item["program_id"],
                "task_id": item["task_id"],
                "family": item["family"],
                "variant": item["variant"],
                "condition": item["condition"],
                "task_text": item["task_text"],
                "reference_text": item["reference_text"],
                "boundary_crossed": item["boundary_crossed"],
                "reference_start_token_index": item["reference_start"],
                **summary,
            }
            append_csv_rows(tokens_path, token_rows)
            append_csv_rows(trajectories_path, [trajectory_row])
        done = len(completed) + offset + len(batch)
        print(
            f"[{done:04d}/{len(prepared)}] batch={len(batch)} "
            f"tokens={token_count} rate={rates[-1]:.1f} tok/s "
            f"elapsed={time.perf_counter() - started:.1f}s"
        )


if __name__ == "__main__":
    main()
