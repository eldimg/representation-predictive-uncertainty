from __future__ import annotations

import argparse
import ast
import collections
import csv
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import (
    AUDIT_FILE,
    CONDITIONS,
    EXPECTED_TASKS_SHA256,
    IR_DERIVED_CONDITIONS,
    IR_FILE,
    MODEL_REGISTRY_FILE,
    ORDER_SEED,
    RENDERED_FILE,
    TASKS_FILE,
    VALIDATION_FILE,
    atomic_write_json,
    build_prompt,
    is_punctuation_only,
    is_whitespace_only,
    load_json,
    map_token_to_semantic_step,
    sha256_file,
    sha256_json,
    write_csv,
)
from differential import run_differential
from ir import dict_to_ast
from opaque_renamer import normalized_ast_equivalent


REQUIRED_AUDIT_CONSTRUCTS = {
    "while": ast.While,
    "break": ast.Break,
    "boolean_expression": ast.BoolOp,
    "dict": ast.Dict,
    "attribute_or_method": ast.Attribute,
}


def _validate_spans(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    text = record["reference_text"]
    spans = sorted(
        record["semantic_step_spans"], key=lambda span: (span["start_char"], span["end_char"])
    )
    previous_end = -1
    seen_steps: set[str] = set()
    for span in spans:
        start = int(span["start_char"])
        end = int(span["end_char"])
        step = str(span["semantic_step_id"])
        if not (0 <= start < end <= len(text)):
            errors.append(f"invalid semantic span {step}: {start}:{end}/{len(text)}")
        if start < previous_end:
            errors.append(f"overlapping semantic span {step}")
        if step in seen_steps:
            errors.append(f"duplicate semantic step {step}")
        previous_end = max(previous_end, end)
        seen_steps.add(step)
    for span in record["trace_spans"]:
        start = int(span["start_char"])
        end = int(span["end_char"])
        if not (0 <= start < end <= len(text)):
            errors.append(f"invalid trace span {span['trace_node_id']}")
    return errors


def _nested_control_flow(tree: ast.AST) -> bool:
    compound = (ast.For, ast.While, ast.If)
    for node in ast.walk(tree):
        if isinstance(node, compound):
            for child in ast.iter_child_nodes(node):
                if child is not node and isinstance(child, compound):
                    return True
                if any(isinstance(grandchild, compound) for grandchild in ast.walk(child)):
                    return True
    return False


def _constructs_by_task(ir_payload: dict[str, Any]) -> dict[str, set[str]]:
    output: dict[str, set[str]] = collections.defaultdict(set)
    for program in ir_payload["programs"]:
        tree = dict_to_ast(program["ast"])
        for label, node_type in REQUIRED_AUDIT_CONSTRUCTS.items():
            if any(isinstance(node, node_type) for node in ast.walk(tree)):
                output[program["task_id"]].add(label)
        if _nested_control_flow(tree):
            output[program["task_id"]].add("nested_control_flow")
        output[program["task_id"]].add("function_interface")
    return output


def _select_audit_tasks(ir_payload: dict[str, Any]) -> tuple[list[str], dict[str, list[str]]]:
    task_ids = sorted({program["task_id"] for program in ir_payload["programs"]})
    random_tasks = sorted(random.Random(ORDER_SEED).sample(task_ids, 20))
    by_task = _constructs_by_task(ir_payload)
    required = set(REQUIRED_AUDIT_CONSTRUCTS) | {
        "nested_control_flow",
        "function_interface",
    }
    covered = set().union(*(by_task[task] for task in random_tasks))
    selected_extra: list[str] = []
    missing = required - covered
    while missing:
        candidates = [task for task in task_ids if task not in random_tasks + selected_extra]
        best = max(candidates, key=lambda task: (len(by_task[task] & missing), task))
        gained = by_task[best] & missing
        if not gained:
            raise RuntimeError(f"No task covers audit constructs: {sorted(missing)}")
        selected_extra.append(best)
        covered |= by_task[best]
        missing = required - covered
    coverage = {
        construct: sorted(
            task
            for task in random_tasks + selected_extra
            if construct in by_task[task]
        )
        for construct in sorted(required)
    }
    return random_tasks + selected_extra, coverage


def _write_audit(
    selected_tasks: list[str],
    records: list[dict[str, Any]],
) -> None:
    selected = set(selected_tasks)
    rows = []
    for record in records:
        if record["task_id"] not in selected:
            continue
        rows.append(
            {
                "task_id": record["task_id"],
                "family": record["family"],
                "variant": record["variant"],
                "condition": record["condition"],
                "task_text": record["task_text"],
                "reference_text": record["reference_text"],
                "source_python": record["source_python"],
                "opaque_mapping": json.dumps(
                    (record.get("opacity") or {}).get("mapping"), ensure_ascii=False
                ),
                "semantic_step_count": len(record["semantic_step_spans"]),
                "validation_flags": "pending_human_review",
            }
        )
    fieldnames = list(rows[0])
    write_csv(AUDIT_FILE, rows, fieldnames)


def _tokenizer_coverage(
    records: list[dict[str, Any]],
    registry: dict[str, Any],
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    output: dict[str, Any] = {}
    for model in registry["models"]:
        tokenizer = AutoTokenizer.from_pretrained(
            model["model_id"],
            revision=model["revision"],
            use_fast=True,
            local_files_only=True,
            trust_remote_code=False,
        )
        if not getattr(tokenizer, "is_fast", False):
            raise RuntimeError(f"Fast tokenizer required for {model['model_id']}")
        aggregates: dict[str, collections.Counter[str]] = {
            condition: collections.Counter() for condition in IR_DERIVED_CONDITIONS
        }
        trajectory_fractions: dict[str, list[float]] = collections.defaultdict(list)
        for record in records:
            condition = record["condition"]
            if condition not in IR_DERIVED_CONDITIONS:
                continue
            base = build_prompt(record["task_text"])
            full = base + record["reference_text"]
            encoding = tokenizer(
                full,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            offsets = list(encoding["offset_mapping"])
            reference_start = len(base)
            first_reference_index = next(
                index for index, (_, end) in enumerate(offsets) if end > reference_start
            )
            counter = aggregates[condition]
            trajectory = collections.Counter()
            spans = record["semantic_step_spans"]
            for token_index, (start, end) in enumerate(offsets):
                if token_index <= first_reference_index or end <= reference_start:
                    continue
                relative_start = max(0, start - reference_start)
                relative_end = end - reference_start
                piece = full[start:end]
                counter["all_analyzed"] += 1
                trajectory["all_analyzed"] += 1
                if is_whitespace_only(piece):
                    counter["whitespace_only"] += 1
                    trajectory["whitespace_only"] += 1
                    continue
                if is_punctuation_only(piece):
                    counter["punctuation_only"] += 1
                    trajectory["punctuation_only"] += 1
                    continue
                counter["eligible"] += 1
                trajectory["eligible"] += 1
                status, _ = map_token_to_semantic_step(
                    relative_start, relative_end, spans
                )
                counter[status] += 1
                trajectory[status] += 1
            fraction = (
                trajectory["mapped"] / trajectory["eligible"]
                if trajectory["eligible"]
                else 0.0
            )
            trajectory_fractions[condition].append(fraction)

        condition_summary = {}
        for condition, counter in aggregates.items():
            eligible = counter["eligible"]
            mapped_fraction = counter["mapped"] / eligible if eligible else 0.0
            all_fraction = (
                counter["mapped"] / counter["all_analyzed"]
                if counter["all_analyzed"]
                else 0.0
            )
            condition_summary[condition] = {
                **dict(counter),
                "step_mapped_token_fraction": mapped_fraction,
                "boundary_overlap_token_fraction": (
                    counter["boundary_overlap"] / eligible if eligible else 0.0
                ),
                "unmapped_token_fraction": (
                    counter["unmapped"] / eligible if eligible else 0.0
                ),
                "step_mapped_token_fraction_all_analyzed": all_fraction,
                "minimum_trajectory_fraction": min(trajectory_fractions[condition]),
                "passes_0_90_gate": mapped_fraction >= 0.90,
            }
        output[model["slug"]] = {
            "model_id": model["model_id"],
            "revision": model["revision"],
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_size": len(tokenizer),
            "conditions": condition_summary,
            "all_conditions_pass": all(
                value["passes_0_90_gate"] for value in condition_summary.values()
            ),
        }
    return output


def validate(*, run_tokenizers: bool = True, run_execution: bool = True) -> dict[str, Any]:
    tasks = load_json(TASKS_FILE)
    ir_payload = load_json(IR_FILE)
    rendered_payload = load_json(RENDERED_FILE)
    records = rendered_payload["records"]
    registry = load_json(MODEL_REGISTRY_FILE)
    errors: list[str] = []

    if sha256_file(TASKS_FILE) != EXPECTED_TASKS_SHA256:
        errors.append("tasks SHA256 mismatch")
    if len(tasks) != 100:
        errors.append(f"task count is {len(tasks)}, expected 100")
    if len({task["family"] for task in tasks}) != 50:
        errors.append("family count is not 50")
    if len(ir_payload["programs"]) != 300:
        errors.append("ProgramIR count is not 300")
    if len(records) != 1800:
        errors.append("rendered trajectory count is not 1800")

    counts = collections.Counter(
        (record["task_id"], record["condition"]) for record in records
    )
    for task in tasks:
        for condition in CONDITIONS:
            if counts[(task["id"], condition)] != 3:
                errors.append(f"{task['id']}/{condition}: expected 3 variants")

    task_lookup = {task["id"]: task for task in tasks}
    record_lookup = {record["trajectory_id"]: record for record in records}
    ir_lookup = {program["program_id"]: program for program in ir_payload["programs"]}
    differential_payloads = []
    span_error_count = 0
    for record in records:
        span_errors = _validate_spans(record)
        if span_errors:
            span_error_count += len(span_errors)
            errors.extend(
                f"{record['trajectory_id']}: {message}" for message in span_errors[:3]
            )
        task = task_lookup[record["task_id"]]
        variant_index = int(record["variant"]) - 1
        if record["condition"] == "nl_exp7_anchor":
            if record["reference_text"] != task["nl"][variant_index]:
                errors.append(f"{record['trajectory_id']}: NL anchor changed")
            if record["semantic_step_spans"]:
                errors.append(f"{record['trajectory_id']}: NL anchor has semantic spans")
        if record["condition"] == "python_canonical":
            if record["reference_text"] != task["python"][variant_index]:
                errors.append(f"{record['trajectory_id']}: canonical Python changed")

    for program_id, program in ir_lookup.items():
        canonical = record_lookup[f"{program_id}__python_canonical"]
        opaque = record_lookup[f"{program_id}__python_opaque"]
        mapping = opaque["opacity"]["mapping"]
        try:
            compile(canonical["reference_text"], "<canonical>", "exec")
            compile(opaque["reference_text"], "<opaque>", "exec")
            if not normalized_ast_equivalent(
                canonical["reference_text"], opaque["reference_text"], mapping
            ):
                errors.append(f"{program_id}: normalized AST mismatch")
        except Exception as error:
            errors.append(f"{program_id}: compile/AST validation: {error}")
        differential_payloads.append(
            {
                "program_id": program_id,
                "task_id": program["task_id"],
                "canonical_source": canonical["reference_text"],
                "opaque_source": opaque["reference_text"],
                "canonical_function_name": program["interface"]["function_name"],
                "opaque_function_name": opaque["opacity"]["opaque_function_name"],
            }
        )

    differential_results = []
    if run_execution:
        differential_results = run_differential(differential_payloads)
        failed = [row for row in differential_results if not row["passed"]]
        if failed:
            errors.extend(
                f"{row['program_id']}: differential execution failed"
                for row in failed[:20]
            )

    selected_tasks, construct_coverage = _select_audit_tasks(ir_payload)
    _write_audit(selected_tasks, records)

    tokenizer_coverage = {}
    if run_tokenizers:
        tokenizer_coverage = _tokenizer_coverage(records, registry)
        for slug, result in tokenizer_coverage.items():
            if not result["all_conditions_pass"]:
                failed_conditions = [
                    condition
                    for condition, value in result["conditions"].items()
                    if not value["passes_0_90_gate"]
                ]
                errors.append(
                    f"{slug}: semantic-step coverage below 0.90 for {failed_conditions}"
                )

    report = {
        "experiment": "Experiment 8 Sequential Representational Interventions",
        "schema_version": "2.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": not errors,
        "errors": errors,
        "counts": {
            "tasks": len(tasks),
            "families": len({task["family"] for task in tasks}),
            "programs": len(ir_payload["programs"]),
            "rendered_trajectories": len(records),
            "conditions": len({record["condition"] for record in records}),
            "span_errors": span_error_count,
            "differential_pairs": len(differential_results),
            "differential_passed": sum(
                int(row["passed"]) for row in differential_results
            ),
        },
        "hashes": {
            "tasks_file_sha256": sha256_file(TASKS_FILE),
            "ir_file_sha256": sha256_file(IR_FILE),
            "ir_content_sha256": ir_payload["content_sha256"],
            "rendered_file_sha256": sha256_file(RENDERED_FILE),
            "rendered_content_sha256": rendered_payload["content_sha256"],
        },
        "audit": {
            "random_seed": ORDER_SEED,
            "selected_task_count": len(selected_tasks),
            "selected_tasks": selected_tasks,
            "construct_coverage": construct_coverage,
            "audit_file_sha256": sha256_file(AUDIT_FILE),
        },
        "differential_execution": {
            "enabled": run_execution,
            "passed": bool(differential_results)
            and all(row["passed"] for row in differential_results),
            "failed_programs": [
                row["program_id"] for row in differential_results if not row["passed"]
            ],
            "total_cases": sum(row.get("case_count", 0) for row in differential_results),
        },
        "tokenizer_coverage": tokenizer_coverage,
    }
    report["report_content_sha256"] = sha256_json(
        {key: value for key, value in report.items() if key != "generated_at_utc"}
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-tokenizers", action="store_true")
    parser.add_argument("--skip-execution", action="store_true")
    parser.add_argument("--output", default=str(VALIDATION_FILE))
    args = parser.parse_args()
    report = validate(
        run_tokenizers=not args.skip_tokenizers,
        run_execution=not args.skip_execution,
    )
    atomic_write_json(Path(args.output), report)
    print(json.dumps(report["counts"], indent=2))
    print(f"Validation passed: {report['passed']}")
    if report["errors"]:
        print("Errors:")
        for error in report["errors"]:
            print(f"- {error}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
