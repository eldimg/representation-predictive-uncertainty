"""Regenerate RELEASE_MANIFEST.json from the current repository snapshot."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "RELEASE_MANIFEST.json"
SOURCE_COMMIT = ROOT / "SOURCE_COMMIT.txt"
SKIP_PARTS = {".git", ".idea", ".vscode", ".venv", "__pycache__"}


def skipped(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return (
        path == OUTPUT
        or any(part in SKIP_PARTS or part.startswith(".venv_") for part in relative.parts)
        or path.suffix.lower() in {".pyc", ".pyo"}
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    source_commit = SOURCE_COMMIT.read_text(encoding="utf-8").strip()
    files = []
    total_size = 0

    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or skipped(path):
            continue
        size = path.stat().st_size
        total_size += size
        files.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "size_bytes": size,
                "sha256": sha256_file(path),
            }
        )

    manifest = {
        "schema_version": 1,
        "project_name": "representation-predictive-uncertainty",
        "package_kind": "standalone Git/code and compact-data release",
        "source_commit": source_commit,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": len(files),
        "total_size_bytes": total_size,
        "files": files,
    }
    OUTPUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"RELEASE MANIFEST WRITTEN: {len(files)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
