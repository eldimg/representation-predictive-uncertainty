from __future__ import annotations

import argparse
import collections
from datetime import datetime, timezone

from common import (
    EXPECTED_TASKS_SHA256,
    IR_FILE,
    TASKS_FILE,
    atomic_write_json,
    load_json,
    sha256_file,
    sha256_json,
)
from ir import build_program_ir


def build_ir() -> dict:
    actual_hash = sha256_file(TASKS_FILE)
    if actual_hash != EXPECTED_TASKS_SHA256:
        raise RuntimeError(
            f"Frozen tasks hash mismatch: {actual_hash} != {EXPECTED_TASKS_SHA256}"
        )
    tasks = load_json(TASKS_FILE)
    if len(tasks) != 100:
        raise RuntimeError(f"Expected 100 tasks, found {len(tasks)}")

    programs = []
    construct_counts: collections.Counter[str] = collections.Counter()
    for task in tasks:
        if len(task.get("nl", [])) != 3 or len(task.get("python", [])) != 3:
            raise RuntimeError(f"{task.get('id')}: expected 3 NL and 3 Python variants")
        for variant_index, source in enumerate(task["python"]):
            program = build_program_ir(task, variant_index, source)
            programs.append(program)
            construct_counts.update(
                node["node_type"] for node in program["trace_nodes"]
            )

    if len(programs) != 300:
        raise RuntimeError(f"Expected 300 ProgramIR records, found {len(programs)}")

    payload = {
        "experiment": "Experiment 8 Sequential Representational Interventions",
        "schema_version": "2.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "tasks_sha256": actual_hash,
        "program_count": len(programs),
        "construct_counts": dict(sorted(construct_counts.items())),
        "programs": programs,
    }
    payload["content_sha256"] = sha256_json(
        {key: value for key, value in payload.items() if key != "generated_at_utc"}
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(IR_FILE))
    args = parser.parse_args()
    payload = build_ir()
    output = IR_FILE if args.output == str(IR_FILE) else __import__("pathlib").Path(args.output)
    atomic_write_json(output, payload)
    print(f"Wrote {len(payload['programs'])} programs to {output}")
    print(f"ProgramIR content SHA256: {payload['content_sha256']}")


if __name__ == "__main__":
    main()
