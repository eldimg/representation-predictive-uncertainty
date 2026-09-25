"""Read-only validation for the standalone Paper 1 release project."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "RELEASE_MANIFEST.json"
EVIDENCE_MANIFEST = ROOT / "EVIDENCE_MANIFEST.json"
EXPECTED_CORPUS_SHA256 = (
    "dcdbc196f7dbf394fbf043690c764c470dbbbf9ee333e344c1f62f320a72a164"
)
CORPUS_COPIES = (
    "experiment6_confirmatory/tasks.json",
    "experiment6_direct_logits/tasks.json",
    "experiment7_cross_model/tasks.json",
    "experiment8_causal_decomposition/tasks.json",
)
FORBIDDEN_TOP_LEVEL = {
    "experiment9_predictive_separation_forecast",
    "experiment11_context_sufficiency_gate",
    "experiment12_literary_hypothesis_profile",
    "experiment13_novel_word_reconstruction",
    "experiment14_single_token_semantic_readout_benchmark",
    "experiment15_compression_series",
    "pilots",
    "local_artifacts",
    "runs",
}
LOCAL_PATH_PATTERN = re.compile(r"[A-Za-z]:[\\/]Users[\\/]")
SKIP_PARTS = {".git", ".idea", ".vscode", ".venv", "__pycache__"}


def skipped(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return (
        any(part in SKIP_PARTS or part.startswith(".venv_") for part in relative.parts)
        or path.suffix.lower() in {".pyc", ".pyo"}
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    failures: list[str] = []
    if not MANIFEST.is_file():
        print("RELEASE VALIDATION FAILED\n- missing RELEASE_MANIFEST.json")
        return 1
    if not EVIDENCE_MANIFEST.is_file():
        print("RELEASE VALIDATION FAILED\n- missing EVIDENCE_MANIFEST.json")
        return 1

    manifest = load_json(MANIFEST)
    expected = {entry["path"]: entry for entry in manifest["files"]}
    actual = {
        path.relative_to(ROOT).as_posix(): path
        for path in ROOT.rglob("*")
        if path.is_file()
        and not skipped(path)
        and path.name != MANIFEST.name
    }

    for relative, entry in expected.items():
        path = actual.get(relative)
        if path is None:
            failures.append(f"missing release file: {relative}")
            continue
        if path.stat().st_size != entry["size_bytes"]:
            failures.append(f"size mismatch: {relative}")
        if sha256_file(path) != entry["sha256"]:
            failures.append(f"SHA-256 mismatch: {relative}")

    unlisted = sorted(set(actual) - set(expected))
    for relative in unlisted:
        failures.append(f"unlisted file: {relative}")

    for name in FORBIDDEN_TOP_LEVEL:
        if (ROOT / name).exists():
            failures.append(f"forbidden top-level path present: {name}")

    token_files = sorted(ROOT.rglob("tokens.csv"))
    for path in token_files:
        failures.append(
            "token-level evidence belongs in the external archive: "
            + path.relative_to(ROOT).as_posix()
        )

    for relative in CORPUS_COPIES:
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing frozen corpus copy: {relative}")
        elif sha256_file(path) != EXPECTED_CORPUS_SHA256:
            failures.append(f"frozen corpus checksum mismatch: {relative}")

    for relative, path in actual.items():
        if path.suffix.lower() not in {".md", ".txt", ".json", ".csv", ".py", ".ps1"}:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if LOCAL_PATH_PATTERN.search(content):
            failures.append(f"absolute local user path found: {relative}")

    evidence = load_json(EVIDENCE_MANIFEST)
    if evidence.get("artifact_count") != 45:
        failures.append("evidence manifest must contain 45 artifacts")
    if evidence.get("missing_count") != 0:
        failures.append("evidence manifest records missing source artifacts")

    if failures:
        print("RELEASE VALIDATION FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("RELEASE VALIDATION PASSED")
    print(f"Files: {len(expected)}")
    print(f"Source commit: {manifest['source_commit']}")
    print(f"Evidence artifacts: {evidence['artifact_count']}")
    print("Token-level CSV files are external and manifest-recorded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
