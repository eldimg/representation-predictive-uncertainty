import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.01
RNG_SEED = 20260916
N_SIGNFLIP = 200_000
N_BOOTSTRAP = 100_000


def signflip_test(deltas, n_resamples=N_SIGNFLIP, seed=RNG_SEED):
    x = np.asarray(deltas, dtype=float)
    observed = x.mean()
    rng = np.random.default_rng(seed)
    extreme = 0
    batch_size = 10_000
    remaining = n_resamples
    while remaining:
        n = min(batch_size, remaining)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(n, len(x)))
        means = (signs * x).mean(axis=1)
        extreme += np.count_nonzero(means >= observed)
        remaining -= n
    return (extreme + 1) / (n_resamples + 1)


def bootstrap_ci(deltas, confidence=0.99, n_resamples=N_BOOTSTRAP, seed=RNG_SEED):
    x = np.asarray(deltas, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(x)
    means = np.empty(n_resamples)
    batch_size = 5000
    written = 0
    while written < n_resamples:
        b = min(batch_size, n_resamples - written)
        idx = rng.integers(0, n, size=(b, n))
        means[written:written+b] = x[idx].mean(axis=1)
        written += b
    q = (1 - confidence) / 2
    return float(np.quantile(means, q)), float(np.quantile(means, 1-q))


def analyze_metric(task_wide, metric):
    delta_col = f"delta_{metric}"
    x = task_wide[delta_col].to_numpy(float)
    w = stats.wilcoxon(x, alternative="greater", zero_method="wilcox")
    t = stats.ttest_1samp(x, 0.0, alternative="greater")
    sd = x.std(ddof=1)
    return {
        "metric": metric,
        "n_tasks": len(x),
        "mean_delta_nl_minus_python": float(x.mean()),
        "median_delta": float(np.median(x)),
        "positive_tasks": int((x > 0).sum()),
        "positive_fraction": float((x > 0).mean()),
        "cohens_dz": float(x.mean()/sd) if sd > 0 else math.inf,
        "signflip_p_one_sided": float(signflip_test(x)),
        "bootstrap_99_ci": list(bootstrap_ci(x)),
        "wilcoxon_p_one_sided": float(w.pvalue),
        "paired_t_p_one_sided": float(t.pvalue),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results_direct_hf")
    args = ap.parse_args()

    results = Path(args.results_dir).resolve()
    traj = pd.read_csv(results / "trajectories.csv")

    if len(traj) != 600:
        raise RuntimeError(f"Expected 600 completed trajectories, found {len(traj)}")

    # Task-level inference: average 3 frozen references per representation.
    grouped = (
        traj.groupby(["task_id", "family", "representation"], as_index=False)
        .agg(
            mean_entropy_top20=("mean_entropy_top20", "mean"),
            mean_entropy_full=("mean_entropy_full", "mean"),
            mean_p1=("mean_p1", "mean"),
            mean_top20_mass=("mean_top20_mass", "mean"),
        )
    )

    task = grouped.pivot(
        index=["task_id", "family"],
        columns="representation",
        values=["mean_entropy_top20", "mean_entropy_full", "mean_p1", "mean_top20_mass"],
    )
    task.columns = [f"{a}_{b}" for a, b in task.columns]
    task = task.reset_index()

    task["delta_mean_entropy_top20"] = (
        task["mean_entropy_top20_nl"] - task["mean_entropy_top20_python"]
    )
    task["delta_mean_entropy_full"] = (
        task["mean_entropy_full_nl"] - task["mean_entropy_full_python"]
    )

    task.to_csv(results / "task_deltas.csv", index=False)

    output = {
        "backend": "direct_hf_bfloat16",
        "top20_comparable": analyze_metric(task, "mean_entropy_top20"),
        "full_vocab_exact": analyze_metric(task, "mean_entropy_full"),
        "mean_entropy_top20_nl": float(task["mean_entropy_top20_nl"].mean()),
        "mean_entropy_top20_python": float(task["mean_entropy_top20_python"].mean()),
        "mean_entropy_full_nl": float(task["mean_entropy_full_nl"].mean()),
        "mean_entropy_full_python": float(task["mean_entropy_full_python"].mean()),
    }

    (results / "analysis_summary.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
