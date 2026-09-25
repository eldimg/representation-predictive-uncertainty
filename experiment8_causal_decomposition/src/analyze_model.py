from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import ANALYSIS_SEED, MODEL_REGISTRY_FILE, atomic_write_json, load_json


N_SIGNFLIP = 200_000
N_BOOTSTRAP = 100_000
CONFIDENCE = 0.99

CONTRASTS = {
    "C1": ("procedural_prose", "controlled_nl", "positive"),
    "C2": ("controlled_nl", "pseudocode", "positive"),
    "C3": ("pseudocode", "python_canonical", "positive"),
    "C4": ("python_opaque", "python_canonical", "two_sided"),
}

METRICS = {
    "h20": "mean_entropy_top20",
    "valid_vocab": "mean_entropy_valid_vocab",
    "semantic_step": "semantic_step_balanced_entropy_top20",
}


def signflip_test(
    values: np.ndarray,
    *,
    two_sided: bool,
    seed: int = ANALYSIS_SEED,
) -> dict[str, Any]:
    x = np.asarray(values, dtype=float)
    observed = float(x.mean())
    rng = np.random.default_rng(seed)
    extreme = 0
    remaining = N_SIGNFLIP
    while remaining:
        count = min(10_000, remaining)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(count, len(x)))
        means = (signs * x).mean(axis=1)
        if two_sided:
            extreme += int(np.count_nonzero(np.abs(means) >= abs(observed)))
        else:
            extreme += int(np.count_nonzero(means >= observed))
        remaining -= count
    return {
        "observed_mean": observed,
        "alternative": "two-sided" if two_sided else "greater",
        "extreme_resamples": extreme,
        "n_resamples": N_SIGNFLIP,
        "p_plus1": (extreme + 1) / (N_SIGNFLIP + 1),
    }


def bootstrap_ci(values: np.ndarray, *, seed: int = ANALYSIS_SEED) -> list[float]:
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(N_BOOTSTRAP, dtype=float)
    written = 0
    while written < N_BOOTSTRAP:
        count = min(5_000, N_BOOTSTRAP - written)
        indices = rng.integers(0, len(x), size=(count, len(x)))
        means[written : written + count] = x[indices].mean(axis=1)
        written += count
    alpha = (1.0 - CONFIDENCE) / 2.0
    return [
        float(np.quantile(means, alpha)),
        float(np.quantile(means, 1.0 - alpha)),
    ]


def cluster_bootstrap_ci(family_means: np.ndarray, *, seed: int = ANALYSIS_SEED) -> list[float]:
    return bootstrap_ci(np.asarray(family_means, dtype=float), seed=seed)


def exact_sign_p(n_positive: int, n_total: int, *, two_sided: bool) -> float:
    upper = sum(math.comb(n_total, k) for k in range(n_positive, n_total + 1)) / (
        2**n_total
    )
    if not two_sided:
        return float(upper)
    lower_count = min(n_positive, n_total - n_positive)
    lower = sum(math.comb(n_total, k) for k in range(0, lower_count + 1)) / (
        2**n_total
    )
    return float(min(1.0, 2.0 * lower))


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        candidate = min(1.0, (total - rank) * value)
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def summarize_values(values: np.ndarray, *, two_sided: bool) -> dict[str, Any]:
    x = np.asarray(values, dtype=float)
    sd = float(x.std(ddof=1))
    positives = int(np.count_nonzero(x > 0))
    negatives = int(np.count_nonzero(x < 0))
    signflip = signflip_test(x, two_sided=two_sided)
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "sd": sd,
        "cohens_dz": float(x.mean() / sd) if sd > 0 else None,
        "positive": positives,
        "negative": negatives,
        "zero": int(len(x) - positives - negatives),
        "positive_fraction": positives / len(x),
        "exact_sign_p": exact_sign_p(positives, len(x), two_sided=two_sided),
        "signflip": signflip,
        "bootstrap_99_ci": bootstrap_ci(x),
    }


def model_margin(model_id: str) -> float:
    registry = load_json(MODEL_REGISTRY_FILE)
    for model in registry["models"]:
        if model["model_id"] == model_id:
            return float(model["c4_equivalence_margin"])
    raise KeyError(f"No C4 margin for {model_id}")


def build_contrasts(trajectories: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = list(METRICS.values())
    grouped = (
        trajectories.groupby(
            ["task_id", "family", "condition"], as_index=False
        )[required]
        .mean()
    )
    rows = []
    for contrast, (left, right, _) in CONTRASTS.items():
        left_frame = grouped[grouped["condition"] == left].set_index(
            ["task_id", "family"]
        )
        right_frame = grouped[grouped["condition"] == right].set_index(
            ["task_id", "family"]
        )
        if set(left_frame.index) != set(right_frame.index):
            raise RuntimeError(f"Task mismatch for {contrast}")
        for index in sorted(left_frame.index):
            task_id, family = index
            row = {
                "contrast": contrast,
                "left_condition": left,
                "right_condition": right,
                "task_id": task_id,
                "family": family,
            }
            for short, column in METRICS.items():
                row[f"delta_{short}"] = (
                    float(left_frame.loc[index, column])
                    - float(right_frame.loc[index, column])
                )
            rows.append(row)
    task = pd.DataFrame(rows)
    family = (
        task.groupby(
            ["contrast", "left_condition", "right_condition", "family"],
            as_index=False,
        )[[f"delta_{short}" for short in METRICS]]
        .mean()
    )
    return task, family


def analyze(results_dir: Path) -> dict[str, Any]:
    trajectories = pd.read_csv(results_dir / "trajectories.csv")
    config = load_json(results_dir / "config.json")
    if len(trajectories) != 1800:
        raise RuntimeError(f"Expected 1800 trajectories, found {len(trajectories)}")
    if trajectories["trajectory_id"].nunique() != 1800:
        raise RuntimeError("Trajectory IDs are not unique")
    if not trajectories["trajectory_valid"].astype(bool).all():
        raise RuntimeError("At least one trajectory is invalid")
    counts = trajectories.groupby(["task_id", "condition"]).size()
    if len(counts) != 600 or not (counts == 3).all():
        raise RuntimeError("Expected exactly 3 variants/task/condition")
    for condition in ("procedural_prose", "controlled_nl", "pseudocode", "python_canonical", "python_opaque"):
        values = trajectories.loc[
            trajectories["condition"] == condition,
            "step_mapped_token_fraction",
        ]
        if values.isna().any() or float(values.mean()) < 0.90:
            raise RuntimeError(f"Semantic-step coverage invalid for {condition}")

    task, family = build_contrasts(trajectories)
    task.to_csv(results_dir / "task_contrasts.csv", index=False)
    family.to_csv(results_dir / "family_contrasts.csv", index=False)

    raw_p: dict[str, float] = {}
    contrast_summaries: dict[str, Any] = {}
    for contrast, (_, _, alternative) in CONTRASTS.items():
        task_rows = task[task["contrast"] == contrast]
        family_rows = family[family["contrast"] == contrast]
        two_sided = alternative == "two_sided"
        primary = summarize_values(task_rows["delta_h20"].to_numpy(), two_sided=two_sided)
        raw_p[contrast] = primary["signflip"]["p_plus1"]
        family_values = family_rows["delta_h20"].to_numpy(float)
        family_summary = summarize_values(family_values, two_sided=two_sided)
        family_summary["cluster_bootstrap_99_ci"] = cluster_bootstrap_ci(family_values)
        valid_mean = float(task_rows["delta_valid_vocab"].mean())
        step_mean = float(task_rows["delta_semantic_step"].mean())
        contrast_summaries[contrast] = {
            "left_condition": CONTRASTS[contrast][0],
            "right_condition": CONTRASTS[contrast][1],
            "alternative": alternative,
            "primary_h20": primary,
            "valid_vocab_mean_delta": valid_mean,
            "semantic_step_balanced_mean_delta": step_mean,
            "family_cluster_robustness": family_summary,
        }

    adjusted = holm_adjust(raw_p)
    for contrast, summary in contrast_summaries.items():
        primary = summary["primary_h20"]
        family_summary = summary["family_cluster_robustness"]
        primary["holm_adjusted_p"] = adjusted[contrast]
        ci = primary["bootstrap_99_ci"]
        if contrast != "C4":
            primary_pass = (
                primary["mean"] > 0
                and adjusted[contrast] < 0.01
                and ci[0] > 0
                and primary["positive"] >= 70
                and family_summary["positive"] >= 35
            )
            if not primary_pass:
                status = "NOT SUPPORTED"
            else:
                robustness_matches = (
                    summary["valid_vocab_mean_delta"] > 0
                    and summary["semantic_step_balanced_mean_delta"] > 0
                )
                status = "ROBUST" if robustness_matches else "METRIC-SENSITIVE"
            summary["primary_pass"] = primary_pass
            summary["result_status"] = status
        else:
            difference_significant = bool(
                adjusted[contrast] < 0.01 and (ci[0] > 0 or ci[1] < 0)
            )
            margin = model_margin(config["model"])
            equivalent = bool(ci[0] > -margin and ci[1] < margin)
            if difference_significant and equivalent:
                interpretation = "statistically_detectable_but_practically_small"
            elif difference_significant:
                interpretation = "statistically_different_potentially_meaningful"
            elif equivalent:
                interpretation = "evidence_for_practical_equivalence"
            else:
                interpretation = "inconclusive"
            primary_direction = np.sign(primary["mean"])
            direction_matches = None
            task_sign_count = (
                primary["positive"] if primary_direction > 0 else primary["negative"]
            )
            family_sign_count = (
                family_summary["positive"]
                if primary_direction > 0
                else family_summary["negative"]
            )
            primary_nonzero_pass = bool(
                difference_significant
                and task_sign_count >= 70
                and family_sign_count >= 35
            )
            status = "NOT SUPPORTED"
            if primary_nonzero_pass:
                direction_matches = bool(
                    np.sign(summary["valid_vocab_mean_delta"]) == primary_direction
                    and np.sign(summary["semantic_step_balanced_mean_delta"])
                    == primary_direction
                )
                status = "ROBUST" if direction_matches else "METRIC-SENSITIVE"
            elif equivalent:
                status = "ROBUST"
            summary.update(
                {
                    "difference_significant": difference_significant,
                    "primary_nonzero_pass": primary_nonzero_pass,
                    "task_sign_count_in_observed_direction": task_sign_count,
                    "family_sign_count_in_observed_direction": family_sign_count,
                    "equivalence_margin": margin,
                    "equivalent": equivalent,
                    "c4_interpretation": interpretation,
                    "directional_robustness_matches": direction_matches,
                    "result_status": status,
                }
            )

    output = {
        "experiment": "Experiment 8 Sequential Representational Interventions",
        "model": config["model"],
        "model_slug": config["model_slug"],
        "model_commit_hash": config["model_commit_hash"],
        "n_trajectories": len(trajectories),
        "n_tasks": int(trajectories["task_id"].nunique()),
        "n_families": int(trajectories["family"].nunique()),
        "analysis_seed": ANALYSIS_SEED,
        "signflip_resamples": N_SIGNFLIP,
        "bootstrap_resamples": N_BOOTSTRAP,
        "holm_family": list(CONTRASTS),
        "contrasts": contrast_summaries,
        "guardrail": (
            "C1-C3 are sequential representational intervention effects, not "
            "isolated single-feature causal effects."
        ),
    }
    atomic_write_json(results_dir / "analysis_summary.json", output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()
    output = analyze(Path(args.results_dir).resolve())
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
