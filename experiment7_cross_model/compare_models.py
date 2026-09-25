import argparse
import json
from pathlib import Path

import pandas as pd


def load_exp7_summary(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    p = data["primary_top20"]
    f = data["family_robustness_primary"]
    c = data["frozen_evidence_criteria_primary"]
    return {
        "source": str(path),
        "model": data.get("model"),
        "model_type": data.get("model_type"),
        "dtype": data.get("dtype"),
        "mean_H20_nl": p["mean_entropy_nl"],
        "mean_H20_python": p["mean_entropy_python"],
        "delta_H20_nl_minus_python": p["mean_delta_nl_minus_python"],
        "positive_tasks": p["positive_tasks"],
        "positive_families": f["positive_families"],
        "cohens_dz": p["cohens_dz"],
        "bootstrap99_low": p["bootstrap_99_ci"][0],
        "bootstrap99_high": p["bootstrap_99_ci"][1],
        "signflip_p": p["signflip_monte_carlo"]["p_one_sided_plus1"],
        "all_four_criteria_pass": c["all_four_pass"],
        "note": "Experiment 7",
    }


def load_qwen_baseline(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    p = data["top20_comparable"]
    return {
        "source": str(path),
        "model": "Qwen/Qwen3-4B",
        "model_type": "qwen3",
        "dtype": "bfloat16",
        "mean_H20_nl": data["mean_entropy_top20_nl"],
        "mean_H20_python": data["mean_entropy_top20_python"],
        "delta_H20_nl_minus_python": p["mean_delta_nl_minus_python"],
        "positive_tasks": p["positive_tasks"],
        "positive_families": None,
        "cohens_dz": p["cohens_dz"],
        "bootstrap99_low": p["bootstrap_99_ci"][0],
        "bootstrap99_high": p["bootstrap_99_ci"][1],
        "signflip_p": p["signflip_p_one_sided"],
        "all_four_criteria_pass": True,
        "note": "Experiment 6 Qwen BF16 baseline",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="*", default=[])
    ap.add_argument("--baseline-qwen", default=None)
    ap.add_argument("--out-dir", default="cross_model_comparison")
    args = ap.parse_args()

    rows = []
    if args.baseline_qwen:
        rows.append(load_qwen_baseline(Path(args.baseline_qwen).resolve()))

    for result_dir in args.results:
        path = Path(result_dir).resolve() / "analysis_summary.json"
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append(load_exp7_summary(path))

    if not rows:
        raise RuntimeError("Provide --results and/or --baseline-qwen.")

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "cross_model_summary.csv", index=False)

    payload = {
        "models": rows,
        "guardrail": (
            "Do not rank raw entropy magnitudes across tokenizers. "
            "Use within-model delta direction, robustness, CI, and paired effect size."
        ),
        "replication_count_all_four_pass": int(
            sum(bool(r["all_four_criteria_pass"]) for r in rows)
        ),
    }
    (out / "cross_model_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(df.to_string(index=False))
    print(f"\nSaved: {out / 'cross_model_summary.csv'}")
    print(f"Saved: {out / 'cross_model_summary.json'}")


if __name__ == "__main__":
    main()
