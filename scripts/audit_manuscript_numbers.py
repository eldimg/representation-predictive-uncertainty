"""Assert empirical numbers used in the submission manuscript against sources."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "publication" / "representation_predictive_uncertainty"


def load_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def close(actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(f"expected {expected!r}, found {actual!r}")


def sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def main() -> int:
    manifest = load_json(
        "publication/representation_predictive_uncertainty/ARTIFACT_MANIFEST.json"
    )
    if manifest["missing_count"] != 0:
        raise AssertionError("artifact manifest has missing files")
    manifest_paths = {entry["path"] for entry in manifest["artifacts"]}

    required_sources = {
        "experiment6_direct_logits/results_direct_hf/trajectories.csv",
        "experiment6_direct_logits/results_direct_hf/analysis_summary.json",
        "experiment7_cross_model/results/phi3/analysis_summary.json",
        "experiment7_cross_model/results/mistral/analysis_summary.json",
        "experiment8_causal_decomposition/results/qwen3/analysis_summary.json",
        "experiment8_causal_decomposition/results/phi3/analysis_summary.json",
        "experiment8_causal_decomposition/results/mistral/analysis_summary.json",
    }
    missing_from_manifest = required_sources - manifest_paths
    if missing_from_manifest:
        raise AssertionError(
            f"manuscript sources absent from manifest: {sorted(missing_from_manifest)}"
        )

    expected_tasks_hash = (
        "dcdbc196f7dbf394fbf043690c764c470dbbbf9ee333e344c1f62f320a72a164"
    )
    for task_file in (
        "experiment6_confirmatory/tasks.json",
        "experiment6_direct_logits/tasks.json",
        "experiment7_cross_model/tasks.json",
        "experiment8_causal_decomposition/tasks.json",
    ):
        assert sha256(task_file) == expected_tasks_hash
    assert sha256("experiment8_causal_decomposition/stimuli/audit_sample.csv") == (
        "7985105006d457b12b7c3297161aed470d19910e4db3f27bbfafeb7fe1a18d4b"
    )

    with (ROOT / "experiment6_direct_logits/results_direct_hf/trajectories.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        trajectories = list(csv.DictReader(handle))
    assert len(trajectories) == 600
    assert len({row["trajectory_id"] for row in trajectories}) == 600
    assert all(row["trajectory_valid"].lower() == "true" for row in trajectories)

    exp6 = load_json("experiment6_direct_logits/results_direct_hf/analysis_summary.json")
    close(exp6["mean_entropy_top20_nl"], 0.5394998116820673)
    close(exp6["mean_entropy_top20_python"], 0.1790361345646111)
    close(exp6["top20_comparable"]["mean_delta_nl_minus_python"], 0.3604636771174562)
    assert exp6["top20_comparable"]["positive_tasks"] == 100
    assert exp6["top20_comparable"]["bootstrap_99_ci"] == [
        0.32312414677456275,
        0.3977158346571436,
    ]
    close(exp6["full_vocab_exact"]["mean_delta_nl_minus_python"], 0.37870211910937795)
    assert exp6["full_vocab_exact"]["bootstrap_99_ci"] == [
        0.3370489523358989,
        0.4204365401369149,
    ]

    replication_expected = {
        "phi3": (0.6592230949553717, [0.60793946624353, 0.7108540012023831]),
        "mistral": (0.6352913522776019, [0.5830957766989548, 0.6869558730665305]),
    }
    for model, (mean, interval) in replication_expected.items():
        summary = load_json(
            f"experiment7_cross_model/results/{model}/analysis_summary.json"
        )
        primary = summary["primary_top20"]
        close(primary["mean_delta_nl_minus_python"], mean)
        assert primary["bootstrap_99_ci"] == interval
        assert primary["positive_tasks"] == 100
        assert summary["family_robustness_primary"]["positive_families"] == 50

    intervention_expected = {
        "qwen3": {"C1": 0.09990061957899049, "C2": 0.07181430674630923,
                  "C3": 0.18349265655130417, "C4": 0.08987777245897391},
        "phi3": {"C1": 0.04250730268617867, "C2": 0.041732827656329324,
                 "C3": 0.2758162811968684, "C4": 0.20918723533125355},
        "mistral": {"C1": -0.005782633279139039, "C2": 0.09683701669778846,
                    "C3": 0.3730460942439051, "C4": 0.16440740962827083},
    }
    family_positive_expected = {
        "qwen3": {"C1": 45, "C2": 41, "C3": 50, "C4": 49},
        "phi3": {"C1": 39, "C2": 41, "C3": 50, "C4": 50},
        "mistral": {"C1": 25, "C2": 49, "C3": 50, "C4": 50},
    }
    total_trajectories = 0
    for model, contrasts in intervention_expected.items():
        summary = load_json(
            f"experiment8_causal_decomposition/results/{model}/analysis_summary.json"
        )
        assert summary["n_tasks"] == 100
        assert summary["n_families"] == 50
        total_trajectories += summary["n_trajectories"]
        for contrast, expected in contrasts.items():
            close(summary["contrasts"][contrast]["primary_h20"]["mean"], expected)
            assert (
                summary["contrasts"][contrast]["family_cluster_robustness"]["positive"]
                == family_positive_expected[model][contrast]
            )
    assert total_trajectories == 5400
    assert load_json(
        "experiment8_causal_decomposition/results/mistral/analysis_summary.json"
    )["contrasts"]["C1"]["result_status"] == "NOT SUPPORTED"

    manuscript_en = (PACKAGE / "MANUSCRIPT_DRAFT_EN.md").read_text(encoding="utf-8")
    assert f"contains {manifest['artifact_count']} key artifacts" in manuscript_en
    for value in (
        "0.3605",
        "0.6592",
        "0.6353",
        "0.0718",
        "0.0417",
        "0.0968",
        "0.1835",
        "0.2758",
        "0.3730",
        "−0.0058",
        "0.0899",
        "0.2092",
        "0.1644",
    ):
        assert value in manuscript_en, f"English manuscript is missing {value}"
    assert "Experiment 9" not in manuscript_en

    print("MANUSCRIPT NUMBER AUDIT PASSED")
    print(f"Manifested artifacts: {manifest['artifact_count']}")
    print("Checked English submission manuscript against Experiments 6-8 results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
