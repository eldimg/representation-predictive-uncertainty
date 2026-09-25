from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import atomic_write_json, load_json


def compare(result_root: Path) -> dict[str, Any]:
    slugs = ("qwen3", "phi3", "mistral")
    summaries = {
        slug: load_json(result_root / slug / "analysis_summary.json") for slug in slugs
    }
    contrasts: dict[str, Any] = {}
    for contrast in ("C1", "C2", "C3"):
        statuses = {
            slug: summaries[slug]["contrasts"][contrast]["result_status"]
            for slug in slugs
        }
        contrasts[contrast] = {
            "model_statuses": statuses,
            "fully_cross_model_replicated": all(
                summaries[slug]["contrasts"][contrast]["primary_pass"]
                for slug in slugs
            ),
            "fully_cross_model_robust": all(status == "ROBUST" for status in statuses.values()),
            "interpretation": (
                "sequential representational intervention; no isolated "
                "surface-feature attribution"
            ),
        }

    c4 = {slug: summaries[slug]["contrasts"]["C4"] for slug in slugs}
    significant = [slug for slug in slugs if c4[slug]["difference_significant"]]
    directions = {
        1 if c4[slug]["primary_h20"]["mean"] > 0 else -1
        for slug in significant
    }
    replicated_difference = (
        len(significant) == 3
        and len(directions) == 1
        and all(c4[slug]["primary_nonzero_pass"] for slug in slugs)
        and all(c4[slug]["result_status"] == "ROBUST" for slug in slugs)
    )
    replicated_equivalence = all(c4[slug]["equivalent"] for slug in slugs)
    contrasts["C4"] = {
        "model_statuses": {slug: c4[slug]["result_status"] for slug in slugs},
        "model_interpretations": {
            slug: c4[slug]["c4_interpretation"] for slug in slugs
        },
        "cross_model_nonzero_difference_replicated": replicated_difference,
        "cross_model_practical_equivalence": replicated_equivalence,
        "heterogeneous": not replicated_difference and not replicated_equivalence,
    }

    output = {
        "experiment": "Experiment 8 Sequential Representational Interventions",
        "models": {
            slug: {
                "model": summaries[slug]["model"],
                "commit": summaries[slug]["model_commit_hash"],
            }
            for slug in slugs
        },
        "contrasts": contrasts,
    }
    analysis_dir = result_root.parent / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(analysis_dir / "cross_model_summary.json", output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    args = parser.parse_args()
    output = compare(Path(args.results_root).resolve())
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
