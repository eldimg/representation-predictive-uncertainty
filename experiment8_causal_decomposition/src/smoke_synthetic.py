from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import ROOT, atomic_write_json, load_json, sha256_json
from ir import build_program_ir
from renderers import render_program
from run_experiment8 import (
    analyze_batch,
    load_model_entry,
    prepare_sequence,
    summarize_rows,
)


DEBUG_TASKS = ROOT / "debug" / "synthetic_tasks.json"


def build_synthetic_records() -> list[dict]:
    records = []
    for item in load_json(DEBUG_TASKS):
        task = {
            "id": item["id"],
            "family": "synthetic_debug",
            "task": item["task"],
            "nl": [
                "Follow the stated synthetic debug procedure.",
                "Follow the stated synthetic debug procedure.",
                "Follow the stated synthetic debug procedure.",
            ],
            "python": [item["python"], item["python"], item["python"]],
        }
        program = build_program_ir(task, 0, item["python"])
        records.extend(render_program(program))
    if len(records) != 30:
        raise RuntimeError(f"Expected 30 synthetic records, found {len(records)}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-slug", choices=("qwen3", "phi3", "mistral"), required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--confirm-synthetic-smoke", action="store_true")
    args = parser.parse_args()
    if not args.confirm_synthetic_smoke:
        raise SystemExit("Pass --confirm-synthetic-smoke to load a model on synthetic tasks")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for synthetic causal-forward smoke")

    entry = load_model_entry(args.model_slug)
    records = build_synthetic_records()
    tokenizer = AutoTokenizer.from_pretrained(
        entry["model_id"],
        revision=entry["revision"],
        use_fast=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        entry["model_id"],
        revision=entry["revision"],
        dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=True,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    model.eval()
    device = torch.device("cuda:0")
    valid_vocab = len(tokenizer)
    prepared = [prepare_sequence(record, tokenizer) for record in records]
    summaries = []
    for offset in range(0, len(prepared), args.batch_size):
        batch = prepared[offset : offset + args.batch_size]
        for item, rows in analyze_batch(
            batch, model, tokenizer, device, valid_vocab
        ):
            summaries.append(
                {
                    "trajectory_id": item["trajectory_id"],
                    "task_id": item["task_id"],
                    "condition": item["condition"],
                    **summarize_rows(rows, item["condition"]),
                }
            )
    output = {
        "kind": "synthetic_debug_only",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_slug": args.model_slug,
        "model": entry["model_id"],
        "revision": entry["revision"],
        "synthetic_task_count": 5,
        "trajectory_count": len(summaries),
        "all_valid": all(row["trajectory_valid"] for row in summaries),
        "records": summaries,
    }
    output["content_sha256"] = sha256_json(
        {key: value for key, value in output.items() if key != "generated_at_utc"}
    )
    destination = ROOT / "debug" / f"smoke_{args.model_slug}.json"
    atomic_write_json(destination, output)
    print(f"Synthetic smoke passed: {len(summaries)} trajectories")
    print(f"Output: {destination}")


if __name__ == "__main__":
    main()
