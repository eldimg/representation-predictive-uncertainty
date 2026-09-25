# Representation-Dependent Predictive Uncertainty

Reproducibility package for **Representation-Dependent Predictive Uncertainty in Autoregressive Language Models**.

This repository contains the frozen task corpus, analysis code, compact derived results, publication tables and figures, and provenance documentation for the three studies reported in the manuscript. Large token-level evidence files are intentionally kept outside Git and are inventoried with SHA-256 checksums in `EVIDENCE_MANIFEST.json`.

## Scope

- Study 1 / Experiment 6: direct Qwen3-4B natural-language versus Python comparison;
- Study 2 / Experiment 7: Phi-3 and Mistral cross-model replications;
- Study 3 / Experiment 8: sequential representational interventions on all three models.

Later exploratory experiments and unrelated research branches are not part of this repository.

## Start here

- `publication/representation_predictive_uncertainty/MANUSCRIPT_DRAFT_EN.md` — canonical submission manuscript;
- `DATASET_CARD.md` — task corpus composition, provenance, limitations, and attribution;
- `publication/representation_predictive_uncertainty/EVIDENCE_MATRIX.md` — manuscript claims mapped to supporting artifacts;
- `publication/representation_predictive_uncertainty/EXPERIMENT6_MODEL_PROVENANCE.md` — Qwen3 Study 1 model-revision provenance;
- `EVIDENCE_MANIFEST.json` — inventory and SHA-256 checksums for the larger evidence archive;
- `RELEASE_MANIFEST.json` — inventory and SHA-256 checksums for the Git release snapshot.

## Validate manuscript numbers

```bash
python scripts/audit_manuscript_numbers.py
```

This checks the main empirical values in the English manuscript against the included compact result artifacts.

## Rebuild publication tables and figures

Install the analysis-only dependencies and run:

```bash
pip install -r publication/representation_predictive_uncertainty/requirements-analysis.txt
python scripts/build_publication_outputs.py
python scripts/audit_manuscript_numbers.py
python scripts/build_release_manifest.py
python scripts/validate_release.py
```

On Windows, `scripts/rebuild_release.ps1` runs the same four publication/release steps. These commands operate on included result artifacts and do not load model weights.

## Evidence and provenance

The frozen corpus contains 100 procedural tasks in 50 paired families. The same corpus is used across the reported studies; its SHA-256 is recorded in the dataset card and validation scripts.

`SOURCE_COMMIT.txt` records a current upstream `llm_research_bench` snapshot containing the working research sources from which this standalone package was separated. The evidence manifests retain their original extraction provenance. Runtime metadata, model revisions, frozen-input hashes, and analysis outputs are preserved in the experiment directories and manifests. The public release history of this repository is intentionally a compact publication history rather than the full exploratory research history.

A DOI is not required to use or inspect this repository. If a separate immutable archive is deposited later, its persistent identifier can be added without changing the scientific contents of this release.

## Known limitations

- The synthetic task corpus was created specifically for this project with substantial use of ChatGPT; provenance is disclosed in `DATASET_CARD.md` and the corpus author declaration.
- The repository history does not provide an independent timestamp for the chronology of the pre-run design documents.
- The exact Qwen3 revision used for Study 1 was not recorded contemporaneously and was later reconstructed from the matching local Hugging Face cache snapshot; this is documented explicitly.
- No second researcher independently reviewed the full stimulus set; this is treated as a limitation rather than as a completed validation step.
- Token-level evidence is too large for the Git package and is therefore represented here by its manifest and checksums.

## Licenses

- executable research code: MIT License (`LICENSE`);
- task corpus, reference solutions, experimental data, figures, and text: Creative Commons Attribution 4.0 International (`LICENSE-DATA`).

Recommended dataset attribution is provided in `DATASET_CARD.md`.
