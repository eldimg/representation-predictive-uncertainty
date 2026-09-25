from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from common import ROOT, atomic_write_json, sha256_file


EXCLUDED_PARTS = {"__pycache__", "results", "analysis"}
EXCLUDED_NAMES = {"MANIFEST.json"}


def manifest_files() -> list[Path]:
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.name in EXCLUDED_NAMES or path.suffix == ".pyc":
            continue
        files.append(path)
    return sorted(files, key=lambda path: path.relative_to(ROOT).as_posix())


def build_manifest() -> dict:
    files = manifest_files()
    return {
        "experiment": "Experiment 8 Sequential Representational Interventions",
        "kind": "pre-run implementation manifest",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "real_model_results_included": False,
        "file_count": len(files),
        "files": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ],
    }


def main() -> None:
    destination = ROOT / "MANIFEST.json"
    manifest = build_manifest()
    atomic_write_json(destination, manifest)
    print(f"Wrote {manifest['file_count']} hashes to {destination}")


if __name__ == "__main__":
    main()
