# Experiment 7 — Cross-Model Replication

This package reuses the exact frozen Experiment 6 stimuli and runs the same
teacher-forced direct-logits measurement on another causal LM.

Goal:

> Was the Experiment 6 NL-vs-Python entropy effect specific to Qwen3-4B?

## Files

- `tasks.json` — frozen 100-task / 600-trajectory stimuli
- `validate_frozen_stimuli.py` — hash/count check
- `run_cross_model.py` — generic direct-logits runner
- `analyze_model.py` — NumPy/Pandas-only model analysis
- `compare_models.py` — Qwen baseline + cross-model comparison
- `verify_env.py` — CUDA check
- `EXPERIMENT_7_PROTOCOL.md` — frozen protocol
- `baseline_qwen_exp6_summary.json` — frozen Qwen Experiment 6 result
- `baseline_qwen_exp6_config.json` — Qwen Experiment 6 config
- `smoke_test_model.ps1` — Windows helper
- `install_env_windows.ps1` — environment recreation helper

## Use your existing direct environment

You can reuse the `.venv_direct` environment from Experiment 6.

From PowerShell:

```powershell
cd .\experiment7_cross_model
```

If using physical GPU 1:

```powershell
$env:CUDA_VISIBLE_DEVICES="1"
python verify_env.py
python validate_frozen_stimuli.py
```

## 20-trajectory smoke test

```powershell
python run_cross_model.py `
  --model "HF_MODEL_ID" `
  --batch-size 8 `
  --limit 20 `
  --results-dir results\MODEL_SLUG_smoke
```

## Full 600 trajectories

Use a NEW result directory:

```powershell
python run_cross_model.py `
  --model "HF_MODEL_ID" `
  --batch-size 8 `
  --results-dir results\MODEL_SLUG
```

If the model needs repository-specific Python code, add:

```text
--trust-remote-code
```

If the model is gated, authenticate with Hugging Face first.

## Analyze

```powershell
python analyze_model.py --results-dir results\MODEL_SLUG
```

Creates:
- `analysis_summary.json`
- `task_deltas.csv`
- `family_deltas.csv`

No SciPy is used.

## Compare with Qwen baseline

```powershell
python compare_models.py `
  --baseline-qwen .\baseline_qwen_exp6_summary.json `
  --results .\results\MODEL_SLUG `
  --out-dir .\cross_model_comparison
```

For two new models:

```powershell
python compare_models.py `
  --baseline-qwen .\baseline_qwen_exp6_summary.json `
  --results .\results\MODEL_A .\results\MODEL_B `
  --out-dir .\cross_model_comparison
```

## Scientific guardrail

Do not rank models by raw entropy. Tokenizers differ.

Compare the within-model quantity:

`H(NL) - H(Python)`

plus sign consistency, family robustness, CI, and Cohen's dz.

## Improvements over Experiment 6 runner

This runner:
1. accepts arbitrary Hugging Face causal LMs;
2. uses current `dtype=` loading;
3. handles BPE boundary fusion with fast-tokenizer offsets;
4. excludes tokenizer-inaccessible LM-head rows from secondary entropy;
5. records architecture/revision/tokenizer metadata;
6. refuses to run if frozen `tasks.json` changes.
