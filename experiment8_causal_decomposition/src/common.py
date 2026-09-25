from __future__ import annotations

import csv
import hashlib
import json
import os
import string
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TASKS_FILE = ROOT / "tasks.json"
STIMULI_DIR = ROOT / "stimuli"
IR_FILE = STIMULI_DIR / "procedure_ir.json"
RENDERED_FILE = STIMULI_DIR / "rendered_stimuli.json"
ORDER_FILE = STIMULI_DIR / "trajectory_order.json"
VALIDATION_FILE = STIMULI_DIR / "validation_report.json"
AUDIT_FILE = STIMULI_DIR / "audit_sample.csv"
MODEL_REGISTRY_FILE = ROOT / "model_registry.json"

EXPECTED_TASKS_SHA256 = (
    "dcdbc196f7dbf394fbf043690c764c470dbbbf9ee333e344c1f62f320a72a164"
)
ORDER_SEED = 20260916
ANALYSIS_SEED = 20260916
TOP_K = 20

CONDITIONS = (
    "nl_exp7_anchor",
    "procedural_prose",
    "controlled_nl",
    "pseudocode",
    "python_canonical",
    "python_opaque",
)
IR_DERIVED_CONDITIONS = CONDITIONS[1:]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, value: Any, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=indent)
            handle.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def build_prompt(task_text: str) -> str:
    return f"Task:\n{task_text}\n\nSolution:\n"


def is_whitespace_only(text: str) -> bool:
    return bool(text) and text.isspace()


def is_punctuation_only(text: str) -> bool:
    nonspace = [character for character in text if not character.isspace()]
    if not nonspace:
        return False
    return all(
        character in string.punctuation
        or unicodedata.category(character).startswith(("P", "S"))
        for character in nonspace
    )


def map_token_to_semantic_step(
    token_start: int,
    token_end: int,
    spans: list[dict[str, Any]],
) -> tuple[str, str | None]:
    """Return (mapping_status, semantic_step_id).

    A token maps only when its complete character interval is contained inside
    exactly one non-overlapping semantic-step span.
    """
    containers = [
        span
        for span in spans
        if token_start >= int(span["start_char"])
        and token_end <= int(span["end_char"])
    ]
    overlaps = [
        span
        for span in spans
        if token_end > int(span["start_char"])
        and token_start < int(span["end_char"])
    ]
    if len(containers) == 1:
        return "mapped", str(containers[0]["semantic_step_id"])
    if len(containers) > 1 or overlaps:
        return "boundary_overlap", None
    return "unmapped", None


def stable_relpath(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())
