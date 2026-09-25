# Experiment 8 — Pre-run Implementation Report

**Status:** implementation and technical validation complete  
**Real benchmark model runs:** not started  
**Human audit:** pending approval

## Implemented independently

- ProgramIR with separate interface and body;
- stable nested `trace_node_id` provenance;
- non-overlapping `semantic_step_id` partition;
- deterministic procedural-prose, controlled-NL, and pseudocode renderers;
- scope-aware, source-format-preserving identifier-opacity transformation;
- parse, compile, scope, normalized-AST, and differential-execution validation;
- tokenizer-specific semantic-step coverage validation;
- safe full-sequence causal runner with an explicit real-run confirmation gate;
- task-level, family-cluster, Holm, equivalence, and cross-model analysis;
- fixed random plus construct-coverage human audit package;
- five-task synthetic debug set and three-model causal-forward smoke tests;
- checkpoint/resume repair and unit tests.

No source file in Experiments 6-7 was edited.

## Generated counts

```text
tasks:                         100
families:                       50
ProgramIR programs:            300
conditions:                      6
rendered trajectories:       1,800
differential program pairs:    300 / 300 passed
differential input cases:      945
span errors:                     0
human-audit tasks:              21
synthetic smoke tasks:           5
synthetic smoke trajectories:   90 / 90 valid (30/model)
```

The human audit contains 20 fixed-seed random tasks plus one additional task,
`count_primes_up_to`, required to cover the rare `break` construct.

## Semantic-step coverage

Every IR-derived condition passes the preregistered aggregate 0.90 gate for all
three frozen tokenizers.

Observed aggregate mapped fractions range from approximately 0.9904 to 0.9979.
Boundary-overlap fraction is zero after assigning a fused newline/indentation plus
code token to the following step while continuing to exclude whitespace-only and
punctuation-only tokens.

## Synthetic causal-forward smoke

Frozen snapshots:

```text
Qwen/Qwen3-4B                         1cfa9a7208912126459214e8b04321603b3df60c
microsoft/Phi-3-mini-4k-instruct      f39ac1d28e925b323eae81227eaba4464caced4e
mistralai/Mistral-7B-Instruct-v0.3    c170c708c41dac9275d15a8fff4eca08d52bab71
```

All three models loaded locally in BF16 on CUDA and each scored 30 synthetic
trajectories. All 90 were valid; every IR-derived synthetic trajectory produced a
non-null semantic-step-balanced metric.

No entropy from the real 100-task Experiment 8 benchmark was computed.

## Key artifact hashes

```text
tasks.json
dcdbc196f7dbf394fbf043690c764c470dbbbf9ee333e344c1f62f320a72a164

procedure_ir.json file
ae6214ef102ef7188d3959ce14b9e8e3439b70561eca6438d634fda2adcd9bb3

ProgramIR canonical content
0d843e99436915a2bf58e6d18a3871186bef3911c5253e907d9aeeee7ffccb18

rendered_stimuli.json file
0fcc7c0b54f9608c174a75c404344ae01af229ba7c31c5e3f18972b8fb2bbe1e

rendered stimuli canonical content
dd23c7af8c16bc83f1fd22c75d3ec1938fbe6b552f49c8f17c8adc636250cd69

audit_sample.csv
7985105006d457b12b7c3297161aed470d19910e4db3f27bbfafeb7fe1a18d4b

synthetic Qwen smoke content
6013b0f5829c87f77b232fa0058a63014736241a1c8a3cb80ebd66482fb62745

synthetic Phi-3 smoke content
56a7d9860b452a83193feca7c35d164717115a7c0fd6268b41a39934dd6449d2

synthetic Mistral smoke content
e3dc227074168fbc7ddd8c5111cf7e0b205a1100aba17e71f2452f43b90ba899
```

## Remaining stop condition

The implementation is technically ready for freeze review. Before real model
runs, a human must inspect `stimuli/audit_sample.csv`, review the validation
report, and explicitly approve the frozen manifest.
