# Experiment 8 — Sequential Representational Interventions

Status: completed and analyzed on Qwen3-4B, Phi-3-mini-4k-instruct, and Mistral-7B-Instruct-v0.3. The saved study contains 5,400 valid trajectories: 1,800 per model.

This directory contains the frozen task copy, deterministic ProgramIR-derived stimuli, validation artifacts, runner and analysis code, compact results, and tests used for Study 3 of the manuscript.

## Provenance note

`DESIGN.md`, `PREREGISTRATION.md`, `IMPLEMENTATION_REPORT.md`, and `MANIFEST.json` are retained as historical internal planning/implementation records. In particular, `PREREGISTRATION.md` is **not an externally registered preregistration and does not provide an independent timestamp**. Its own saved status is part of the record. The manuscript therefore describes criteria as predefined rather than claiming formal preregistration.

## Rebuild and validation

The deterministic stimulus pipeline is implemented in `src/` and tested in `tests/`. The saved stimuli are under `stimuli/`; model revisions and the C4 equivalence margins are stored in `model_registry.json`.

The compact result artifacts used by the manuscript are under `results/<model>/` and `analysis/`. Large token-level evidence is not stored in Git; its checksums and expected sizes are recorded in the repository-level `EVIDENCE_MANIFEST.json`.

## Main saved outputs

- `analysis/cross_model_summary.json` — cross-model summary;
- `analysis/RESULTS_SUMMARY_RU.md` — historical human-readable results summary;
- `results/<model>/analysis_summary.json` — per-model statistical summaries;
- `results/<model>/trajectories.csv` — saved trajectory-level outputs;
- `results/<model>/task_contrasts.csv` and `family_contrasts.csv` — task/family contrasts.

Publication figures and tables are regenerated from these compact artifacts with `../scripts/build_publication_outputs.py` from the repository root.
