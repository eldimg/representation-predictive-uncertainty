# EXPERIMENT 7 — Cross-Model Replication Protocol

Status: FROZEN BEFORE NON-QWEN RESULTS  
Date: 2026-09-16

## Research question

Does the within-model predictive-entropy separation observed in Experiment 6
(NL > Python) replicate in independently trained causal language models?

## Frozen stimuli

- 100 procedural tasks
- 50 task families
- 3 natural-language references per task
- 3 Python references per task
- 600 trajectories per model
- `tasks.json` SHA256: `dcdbc196f7dbf394fbf043690c764c470dbbbf9ee333e344c1f62f320a72a164`

The stimulus texts MUST NOT be edited for Experiment 7.

## Prompt

Exactly:

```text
Task:
<task>

Solution:
<reference>
```

No chat template. No representation instruction. No sampling.

## Measurement

Teacher-forced direct causal forward pass.

Primary trajectory metric: `mean_entropy_top20`.

The first reference token is excluded, matching Experiment 6.

Secondary: `mean_entropy_valid_vocab`.

For models whose LM head has padded/unused rows, secondary entropy is
normalized only across tokenizer-addressable token ids.

## Inferential unit

Task, N=100.

For each task:
1. average the three NL trajectory means;
2. average the three Python trajectory means;
3. compute `delta_H = H_NL - H_Python`.

Positive delta means lower entropy for Python.

## Frozen evidence criteria per model

A model-level replication passes only if all four hold:

1. one-sided sign-flip p < 0.01
2. 99% bootstrap CI for mean delta entirely above zero
3. at least 70 / 100 tasks positive
4. at least 35 / 50 families positive

## Cross-model interpretation

Do NOT rank raw entropy magnitudes between models. Different tokenizers use
different token units and vocabularies.

The replication target is:
- direction of within-model NL-Python delta
- task-level sign consistency
- family robustness
- confidence interval
- standardized paired effect size

One non-Qwen model passing all four criteria is a first cross-model replication.
Two or more independently trained non-Qwen model families passing them would
provide stronger evidence that the effect is not Qwen-specific.

## What this does not establish

A successful Experiment 7 still does not establish that code is universally
lower entropy or identify the causal property of Python. That belongs to the
next causal-ablation experiment.
