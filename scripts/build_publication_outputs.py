"""Generate publication tables and figures from existing analysis artifacts.

This script performs analysis-only reads. It never loads model weights and never
modifies experiment result directories.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "publication" / "representation_predictive_uncertainty"
OUTPUT = PACKAGE / "generated"

MODEL_LABELS = {
    "qwen3": "Qwen3-4B",
    "phi3": "Phi-3 mini",
    "mistral": "Mistral 7B",
}
MODEL_COLORS = {
    "qwen3": "#2166ac",
    "phi3": "#1b9e77",
    "mistral": "#d95f02",
}
CONTRAST_COLORS = {
    "C1": "#7570b3",
    "C2": "#1b9e77",
    "C3": "#d95f02",
    "C4": "#e7298a",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def build_replication_table() -> list[dict[str, Any]]:
    qwen = load_json(
        ROOT
        / "experiment6_direct_logits"
        / "results_direct_hf"
        / "analysis_summary.json"
    )["top20_comparable"]
    rows = [
        {
            "model_slug": "qwen3",
            "model": MODEL_LABELS["qwen3"],
            "experiment": 6,
            "mean_delta_nl_minus_python": qwen["mean_delta_nl_minus_python"],
            "ci99_low": qwen["bootstrap_99_ci"][0],
            "ci99_high": qwen["bootstrap_99_ci"][1],
            "positive_tasks": qwen["positive_tasks"],
            "total_tasks": qwen["n_tasks"],
            "cohens_dz": qwen["cohens_dz"],
        }
    ]
    for slug in ("phi3", "mistral"):
        summary = load_json(
            ROOT
            / "experiment7_cross_model"
            / "results"
            / slug
            / "analysis_summary.json"
        )["primary_top20"]
        rows.append(
            {
                "model_slug": slug,
                "model": MODEL_LABELS[slug],
                "experiment": 7,
                "mean_delta_nl_minus_python": summary["mean_delta_nl_minus_python"],
                "ci99_low": summary["bootstrap_99_ci"][0],
                "ci99_high": summary["bootstrap_99_ci"][1],
                "positive_tasks": summary["positive_tasks"],
                "total_tasks": summary["n_tasks"],
                "cohens_dz": summary["cohens_dz"],
            }
        )
    return rows


def build_intervention_table() -> list[dict[str, Any]]:
    rows = []
    for slug in ("qwen3", "phi3", "mistral"):
        summary = load_json(
            ROOT
            / "experiment8_causal_decomposition"
            / "results"
            / slug
            / "analysis_summary.json"
        )
        for contrast in ("C1", "C2", "C3", "C4"):
            result = summary["contrasts"][contrast]
            primary = result["primary_h20"]
            row = {
                "model_slug": slug,
                "model": MODEL_LABELS[slug],
                "contrast": contrast,
                "left_condition": result["left_condition"],
                "right_condition": result["right_condition"],
                "mean_delta": primary["mean"],
                "ci99_low": primary["bootstrap_99_ci"][0],
                "ci99_high": primary["bootstrap_99_ci"][1],
                "positive_tasks": primary["positive"],
                "total_tasks": primary["n"],
                "result_status": result["result_status"],
                "valid_vocab_mean_delta": result["valid_vocab_mean_delta"],
                "semantic_step_balanced_mean_delta": result[
                    "semantic_step_balanced_mean_delta"
                ],
            }
            if contrast == "C4":
                row["equivalence_margin"] = result["equivalence_margin"]
                row["equivalent"] = result["equivalent"]
            else:
                row["equivalence_margin"] = ""
                row["equivalent"] = ""
            rows.append(row)
    return rows


def build_condition_table() -> list[dict[str, Any]]:
    conditions = (
        "nl_exp7_anchor",
        "procedural_prose",
        "controlled_nl",
        "pseudocode",
        "python_canonical",
        "python_opaque",
    )
    rows = []
    for slug in ("qwen3", "phi3", "mistral"):
        frame = pd.read_csv(
            ROOT
            / "experiment8_causal_decomposition"
            / "results"
            / slug
            / "trajectories.csv",
            usecols=["condition", "mean_entropy_top20"],
        )
        means = frame.groupby("condition")["mean_entropy_top20"].mean()
        for condition in conditions:
            rows.append(
                {
                    "model_slug": slug,
                    "model": MODEL_LABELS[slug],
                    "condition": condition,
                    "mean_entropy_top20": float(means[condition]),
                }
            )
    return rows


def build_family_table() -> list[dict[str, Any]]:
    qwen_tasks = pd.read_csv(
        ROOT
        / "experiment6_direct_logits"
        / "results_direct_hf"
        / "task_deltas.csv",
        usecols=["family", "delta_mean_entropy_top20"],
    )
    qwen_family = qwen_tasks.groupby("family")["delta_mean_entropy_top20"].mean()
    rows = [
        {
            "experiment": 6,
            "model_slug": "qwen3",
            "model": MODEL_LABELS["qwen3"],
            "contrast": "NL-Python",
            "positive_families": int((qwen_family > 0).sum()),
            "total_families": int(len(qwen_family)),
            "mean_family_delta": float(qwen_family.mean()),
            "min_family_delta": float(qwen_family.min()),
            "max_family_delta": float(qwen_family.max()),
        }
    ]
    for slug in ("phi3", "mistral"):
        summary = load_json(
            ROOT
            / "experiment7_cross_model"
            / "results"
            / slug
            / "analysis_summary.json"
        )
        family = summary["family_robustness_primary"]
        rows.append(
            {
                "experiment": 7,
                "model_slug": slug,
                "model": MODEL_LABELS[slug],
                "contrast": "NL-Python",
                "positive_families": family["positive_families"],
                "total_families": family["total_families"],
                "mean_family_delta": summary["primary_top20"][
                    "mean_delta_nl_minus_python"
                ],
                "min_family_delta": family["min_family_delta"],
                "max_family_delta": family["max_family_delta"],
            }
        )
    for slug in ("qwen3", "phi3", "mistral"):
        summary = load_json(
            ROOT
            / "experiment8_causal_decomposition"
            / "results"
            / slug
            / "analysis_summary.json"
        )
        for contrast in ("C1", "C2", "C3", "C4"):
            family = summary["contrasts"][contrast]["family_cluster_robustness"]
            rows.append(
                {
                    "experiment": 8,
                    "model_slug": slug,
                    "model": MODEL_LABELS[slug],
                    "contrast": contrast,
                    "positive_families": family["positive"],
                    "total_families": family["n"],
                    "mean_family_delta": family["mean"],
                    "min_family_delta": "",
                    "max_family_delta": "",
                }
            )
    return rows


def replication_figure(rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    y = np.arange(len(rows))
    for index, row in enumerate(rows):
        mean = float(row["mean_delta_nl_minus_python"])
        low = float(row["ci99_low"])
        high = float(row["ci99_high"])
        ax.errorbar(
            mean,
            index,
            xerr=[[mean - low], [high - mean]],
            fmt="o",
            color=MODEL_COLORS[str(row["model_slug"])],
            capsize=4,
            markersize=7,
        )
    ax.axvline(0, color="#444444", linewidth=1, linestyle="--")
    ax.set_yticks(y, [str(row["model"]) for row in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Mean task-level ΔH20 (NL − Python), 99% bootstrap CI")
    ax.set_title("NL–Python predictive-entropy effect replicates across models")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT / "figure_1_replication_forest.png", bbox_inches="tight")
    plt.close(fig)


def intervention_figure(rows: list[dict[str, Any]]) -> None:
    ordered = sorted(
        rows,
        key=lambda row: (
            ("C1", "C2", "C3", "C4").index(str(row["contrast"])),
            ("qwen3", "phi3", "mistral").index(str(row["model_slug"])),
        ),
    )
    fig, ax = plt.subplots(figsize=(8.0, 6.8))
    y = np.arange(len(ordered))
    labels = []
    for index, row in enumerate(ordered):
        mean = float(row["mean_delta"])
        low = float(row["ci99_low"])
        high = float(row["ci99_high"])
        ax.errorbar(
            mean,
            index,
            xerr=[[mean - low], [high - mean]],
            fmt="o",
            color=CONTRAST_COLORS[str(row["contrast"])],
            capsize=3,
            markersize=6,
        )
        labels.append(f"{row['contrast']} · {row['model']}")
    ax.axvline(0, color="#444444", linewidth=1, linestyle="--")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean task-level contrast ΔH20, 99% bootstrap CI")
    ax.set_title("Sequential representation interventions")
    ax.grid(axis="x", alpha=0.2)
    for boundary in (2.5, 5.5, 8.5):
        ax.axhline(boundary, color="#dddddd", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(OUTPUT / "figure_3_intervention_forest.png", bbox_inches="tight")
    plt.close(fig)


def condition_figure(rows: list[dict[str, Any]]) -> None:
    labels = {
        "nl_exp7_anchor": "NL anchor",
        "procedural_prose": "Procedural prose",
        "controlled_nl": "Controlled NL",
        "pseudocode": "Pseudocode",
        "python_canonical": "Canonical Python",
        "python_opaque": "Opaque Python",
    }
    order = list(labels)
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    for slug in ("qwen3", "phi3", "mistral"):
        by_condition = {
            str(row["condition"]): float(row["mean_entropy_top20"])
            for row in rows
            if row["model_slug"] == slug
        }
        ax.scatter(
            [0],
            [by_condition["nl_exp7_anchor"]],
            marker="D",
            s=48,
            color=MODEL_COLORS[slug],
            zorder=3,
        )
        ax.plot(
            np.arange(1, len(order)),
            [by_condition[condition] for condition in order[1:]],
            marker="o",
            linewidth=2,
            color=MODEL_COLORS[slug],
            label=MODEL_LABELS[slug],
        )
    ax.set_xticks(np.arange(len(order)), [labels[item] for item in order], rotation=20)
    ax.set_ylabel("Mean top-20 entropy")
    ax.set_title("Predictive uncertainty across representation conditions")
    ax.axvline(0.5, color="#bbbbbb", linewidth=1, linestyle=":")
    ax.text(
        0.25,
        0.97,
        "independent anchor",
        ha="center",
        va="top",
        fontsize=8,
        color="#555555",
        transform=ax.get_xaxis_transform(),
    )
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT / "figure_2_condition_profile.png", bbox_inches="tight")
    plt.close(fig)


def c4_figure(rows: list[dict[str, Any]]) -> None:
    c4 = [row for row in rows if row["contrast"] == "C4"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    y = np.arange(len(c4))
    for index, row in enumerate(c4):
        mean = float(row["mean_delta"])
        low = float(row["ci99_low"])
        high = float(row["ci99_high"])
        margin = float(row["equivalence_margin"])
        ax.barh(
            index,
            2 * margin,
            left=-margin,
            height=0.45,
            color="#dddddd",
            label="Equivalence region" if index == 0 else None,
        )
        ax.errorbar(
            mean,
            index,
            xerr=[[mean - low], [high - mean]],
            fmt="o",
            color=MODEL_COLORS[str(row["model_slug"])],
            capsize=4,
            markersize=7,
        )
    ax.axvline(0, color="#444444", linewidth=1)
    ax.set_yticks(y, [str(row["model"]) for row in c4])
    ax.invert_yaxis()
    ax.set_xlabel("C4 ΔH20 (opaque − canonical), 99% bootstrap CI")
    ax.set_title("Identifier-opacity effects exceed frozen equivalence regions")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTPUT / "figure_4_c4_equivalence.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    configure_plotting()

    replication = build_replication_table()
    interventions = build_intervention_table()
    conditions = build_condition_table()
    families = build_family_table()

    save_csv(OUTPUT / "table_1_replication.csv", replication)
    save_csv(OUTPUT / "table_2_interventions.csv", interventions)
    save_csv(OUTPUT / "table_3_condition_means.csv", conditions)
    save_csv(OUTPUT / "table_4_family_robustness.csv", families)
    replication_figure(replication)
    condition_figure(conditions)
    intervention_figure(interventions)
    c4_figure(interventions)

    print(f"Wrote publication outputs to {OUTPUT}")


if __name__ == "__main__":
    main()
