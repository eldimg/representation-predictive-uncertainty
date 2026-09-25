# Changelog

Notable changes to the standalone reproducibility package are documented here.

## 1.0.0-arxiv - 2026-09-25

- Prepared the repository as the clean public companion to the submission manuscript.
- Finalized the English manuscript as the canonical submission version; working manuscript translations remain in the private research repository rather than this release.
- Aligned publication figures, captions, bibliography, author metadata, and manuscript-number auditing.
- Removed internal drafting/audit notes and an unused Gemma corpus duplicate that are not part of the reported studies.
- Replaced machine-local task paths in the Study 1 configs with repository-relative paths.
- Added a release-manifest builder so the final public snapshot can be checksum-inventoried after regeneration.
- Retained the frozen 100-task corpus, study code, compact results, provenance records, licenses, and the 45-artifact external-evidence inventory.
- Token-level CSV evidence remains outside Git and is checksum-inventoried in `EVIDENCE_MANIFEST.json`.

## 0.1.0-local - 2026-09-24

- Created the initial local dry-run package for Experiments 6–8.
- Added the frozen corpus, deterministic Study 3 stimuli, analysis code, compact results, publication outputs, and provenance records.
