import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ALPHA = 0.01
RNG_SEED = 20260916
N_SIGNFLIP = 200_000
N_BOOTSTRAP = 100_000


def signflip_test(deltas, n_resamples=N_SIGNFLIP, seed=RNG_SEED):
    x = np.asarray(deltas, dtype=float)
    observed = float(x.mean())
    rng = np.random.default_rng(seed)
    extreme = 0
    remaining = n_resamples
    batch = 10_000
    while remaining:
        n = min(batch, remaining)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(n, len(x)))
        means = (signs * x).mean(axis=1)
        extreme += int(np.count_nonzero(means >= observed))
        remaining -= n
    return {
        "observed_mean": observed,
        "extreme_resamples": extreme,
        "n_resamples": n_resamples,
        "p_one_sided_plus1": (extreme + 1) / (n_resamples + 1),
    }


def bootstrap_ci(deltas, confidence=0.99, n_resamples=N_BOOTSTRAP, seed=RNG_SEED):
    x = np.asarray(deltas, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(x)
    means = np.empty(n_resamples, dtype=float)
    written = 0
    batch = 5_000
    while written < n_resamples:
        b = min(batch, n_resamples - written)
        idx = rng.integers(0, n, size=(b, n))
        means[written:written+b] = x[idx].mean(axis=1)
        written += b
    q = (1.0 - confidence) / 2.0
    return [float(np.quantile(means, q)), float(np.quantile(means, 1.0-q))]


def exact_sign_p_one_sided(n_positive, n_total):
    # Under p=0.5, P[X >= n_positive], X~Binomial(n_total, 0.5)
    numerator = sum(math.comb(n_total, k) for k in range(n_positive, n_total + 1))
    return numerator / (2 ** n_total)


def metric_summary(task, delta_col):
    x = task[delta_col].to_numpy(float)
    sd = float(x.std(ddof=1))
    positives = int((x > 0).sum())
    sf = signflip_test(x)
    ci = bootstrap_ci(x)
    return {
        "n_tasks": int(len(x)),
        "mean_delta_nl_minus_python": float(x.mean()),
        "median_delta": float(np.median(x)),
        "sd_delta": sd,
        "cohens_dz": float(x.mean() / sd) if sd > 0 else None,
        "positive_tasks": positives,
        "positive_fraction": float(positives / len(x)),
        "exact_sign_test_p_one_sided": float(exact_sign_p_one_sided(positives, len(x))),
        "signflip_monte_carlo": sf,
        "bootstrap_99_ci": ci,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results_direct_hf")
    args = ap.parse_args()

    results = Path(args.results_dir).resolve()
    traj_file = results / "trajectories.csv"
    traj = pd.read_csv(traj_file)

    if len(traj) != 600:
        raise RuntimeError(f"Expected 600 trajectories, found {len(traj)}")

    if not bool(traj["trajectory_valid"].all()):
        raise RuntimeError("At least one trajectory is invalid.")

    counts = traj.groupby(["task_id", "representation"]).size().unstack(fill_value=0)
    if not ((counts["nl"] == 3) & (counts["python"] == 3)).all():
        raise RuntimeError("Expected exactly 3 NL and 3 Python trajectories per task.")

    metrics = ["mean_entropy_top20", "mean_entropy_full", "mean_p1", "mean_top20_mass"]
    grouped = (
        traj.groupby(["task_id", "family", "representation"], as_index=False)[metrics]
        .mean()
    )
    task = grouped.pivot(
        index=["task_id", "family"],
        columns="representation",
        values=metrics,
    )
    task.columns = [f"{metric}_{rep}" for metric, rep in task.columns]
    task = task.reset_index()

    task["delta_entropy_top20"] = (
        task["mean_entropy_top20_nl"] - task["mean_entropy_top20_python"]
    )
    task["delta_entropy_full"] = (
        task["mean_entropy_full_nl"] - task["mean_entropy_full_python"]
    )

    family = (
        task.groupby("family", as_index=False)
        .agg(
            delta_entropy_top20=("delta_entropy_top20", "mean"),
            delta_entropy_full=("delta_entropy_full", "mean"),
        )
    )

    top20 = metric_summary(task, "delta_entropy_top20")
    full = metric_summary(task, "delta_entropy_full")

    output = {
        "backend": "Qwen/Qwen3-4B BF16 direct Transformers forward",
        "n_trajectories": int(len(traj)),
        "n_tasks": int(task["task_id"].nunique()),
        "n_families": int(task["family"].nunique()),
        "all_trajectories_valid": bool(traj["trajectory_valid"].all()),
        "special_token_events_total": int(traj["special_token_events"].sum()),
        "top20": {
            "mean_nl": float(task["mean_entropy_top20_nl"].mean()),
            "mean_python": float(task["mean_entropy_top20_python"].mean()),
            "relative_reduction_python_vs_nl": float(
                1.0 - task["mean_entropy_top20_python"].mean()
                / task["mean_entropy_top20_nl"].mean()
            ),
            **top20,
        },
        "full_vocab": {
            "mean_nl": float(task["mean_entropy_full_nl"].mean()),
            "mean_python": float(task["mean_entropy_full_python"].mean()),
            "relative_reduction_python_vs_nl": float(
                1.0 - task["mean_entropy_full_python"].mean()
                / task["mean_entropy_full_nl"].mean()
            ),
            **full,
        },
        "mean_p1_nl": float(task["mean_p1_nl"].mean()),
        "mean_p1_python": float(task["mean_p1_python"].mean()),
        "family_robustness_top20": {
            "positive_families": int((family["delta_entropy_top20"] > 0).sum()),
            "total_families": int(len(family)),
            "min_family_delta": float(family["delta_entropy_top20"].min()),
            "max_family_delta": float(family["delta_entropy_top20"].max()),
        },
        "frozen_evidence_criteria_top20": {
            "signflip_p_lt_0_01": bool(top20["signflip_monte_carlo"]["p_one_sided_plus1"] < 0.01),
            "bootstrap_99_ci_entirely_above_zero": bool(top20["bootstrap_99_ci"][0] > 0),
            "at_least_70_of_100_tasks_positive": bool(top20["positive_tasks"] >= 70),
            "at_least_35_of_50_families_positive": bool(
                int((family["delta_entropy_top20"] > 0).sum()) >= 35
            ),
        },
    }

    task.to_csv(results / "task_deltas.csv", index=False)
    family.to_csv(results / "family_deltas.csv", index=False)
    (results / "analysis_summary_numpy.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
