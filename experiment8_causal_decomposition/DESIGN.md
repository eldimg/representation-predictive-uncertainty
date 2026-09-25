# Experiment 8 — Sequential Representational Interventions on Predictive Entropy

**Status:** design v2 / candidate for preregistration freeze  
**Date:** 2026-09-16  
**Supersedes:** `Experiment_8_Causal_Decomposition_Design.md`  
**Implementation state:** not started  
**Real model runs:** prohibited until the freeze checklist is approved

---

## 1. Scientific scope

Experiment 7 established a replicated within-model effect on a frozen benchmark:

```text
H(NL) > H(Python)
```

for all 100 task-level comparisons in each of three model families.

Experiment 8 asks which deterministic changes in representation alter predictive
uncertainty while the represented procedure is held fixed.

Experiment 8 is a causal intervention experiment at the level of complete
representations: a deterministic renderer intervenes on representation while a
shared ProgramIR holds procedural semantics fixed.

It is not a feature-orthogonal factorial experiment. C1-C3 are therefore defined
as **sequential representational intervention effects**, not isolated causal
effects of capitalization, punctuation, indentation, symbolism, syntax, or any
other single surface feature. Each transition changes a bundle of surface
properties.

The canonical sequence is:

```text
procedural_prose
       ↓ C1
controlled_nl
       ↓ C2
pseudocode
       ↓ C3
python_canonical
```

The defensible claim is:

> Under fixed procedural semantics, this deterministic transition between two
> complete representations changes within-model next-token predictive entropy.

The design does not license the stronger claim that one particular surface
feature alone caused the change.

---

## 2. Frozen source dataset

Reuse the exact 100-task / 50-family dataset from Experiments 6-7.

Required `tasks.json` SHA256:

```text
dcdbc196f7dbf394fbf043690c764c470dbbbf9ee333e344c1f62f320a72a164
```

The source contains, per task:

- 3 natural-language references;
- 3 canonical Python references;
- a family label;
- a task statement.

All 300 canonical Python references are complete `FunctionDef` programs. The
source file must not be edited, and no task or variant may be dropped.

---

## 3. Models

Use exactly:

```text
Qwen/Qwen3-4B
microsoft/Phi-3-mini-4k-instruct
mistralai/Mistral-7B-Instruct-v0.3
```

Known closed-run commits:

```text
Qwen3-4B:     1cfa9a7208912126459214e8b04321603b3df60c
Phi-3 Mini:  f39ac1d28e925b323eae81227eaba4464caced4e
Mistral-7B:  c170c708c41dac9275d15a8fff4eca08d52bab71
```

The Experiment 6 Qwen config did not record a commit hash, but pre-run
implementation inspection resolved the locally cached snapshot used by that run
to `1cfa9a7208912126459214e8b04321603b3df60c`. This value is frozen in
`model_registry.json`. A floating `main` revision is not allowed for any real run.

No additional model family may be added after results are inspected.

---

## 4. Representation conditions

Create exactly six conditions and exactly three variants per task per condition.

### 4.1 `nl_exp7_anchor`

The original Experiment 7 NL reference, byte-for-byte unchanged.

Purpose:

- historical and pipeline-continuity anchor;
- sanity replication of the earlier NL-vs-Python contrast.

This condition is not generated from ProgramIR and does not receive
semantic-step-balanced analysis.

### 4.2 `procedural_prose`

A deterministic fluent procedural-English rendering of ProgramIR. It must state
the procedure interface and then describe the body in fixed templates.

Example interface:

```text
Define a procedure named sum_positive that receives values.
```

### 4.3 `controlled_nl`

A deterministic one-operation-per-instruction controlled-English rendering of
the same ProgramIR.

Example interface:

```text
PROCEDURE sum_positive INPUT values:
```

### 4.4 `pseudocode`

A deterministic symbolic pseudocode rendering of the same ProgramIR.

Example interface:

```text
PROCEDURE sum_positive(values)
```

### 4.5 `python_canonical`

The original frozen Python reference, byte-for-byte unchanged.

Example interface:

```python
def sum_positive(values):
```

### 4.6 `python_opaque`

The same Python program after a deterministic, scope-aware
identifier-opacity intervention. Syntax, binding structure, operations, control
flow, and runtime behavior must be preserved.

C4 estimates the effect of the complete identifier-opacity intervention. It is
not interpreted as a pure effect of identifier semantics because opacity can also
change corpus familiarity, morphology, length, and tokenizer segmentation.

---

## 5. ProgramIR

All IR-derived conditions must originate from the same ProgramIR instance for a
given canonical Python variant.

```text
canonical Python
      ↓ deterministic parser
ProgramIR
      ├─ interface
      │    ├─ function_name
      │    └─ parameters[]
      └─ body
           └─ semantic steps and expression provenance
      ↓ deterministic renderers
procedural_prose / controlled_nl / pseudocode / python_opaque
```

Canonical Python remains byte-for-byte unchanged, but its interface and body are
parsed into the same ProgramIR for provenance and semantic-step mapping.

### 5.1 Interface contract

Required structure:

```json
{
  "interface": {
    "function_name": "sum_positive",
    "parameters": ["values"]
  },
  "body": {
    "steps": []
  }
}
```

The interface is a separate representational component, not an ordinary
computational AST statement. All five IR-derived conditions must encode the same
function arity, parameter order, and binding roles. Procedural prose, controlled
NL, pseudocode, and canonical Python preserve the canonical lexical names.
Opaque Python uses its deterministic renamed function and parameter identifiers,
with an explicit reversible mapping back to the ProgramIR interface.

### 5.2 Body coverage

The body representation must support every construct actually present in all 300
frozen programs. At minimum this includes:

```text
ASSIGN
UPDATE
FOR_EACH
WHILE
IF
ELSE
RETURN
APPEND
BREAK
EXPRESSION_STATEMENT
BOOL_OP
CALL
ATTRIBUTE_OR_METHOD_ACCESS
INDEX
SLICE
LIST
TUPLE
DICT
UNARY_OP
BINARY_OP
COMPARE
MEMBERSHIP
CONSTANT
VARIABLE
```

Unsupported syntax must fail validation. It may not be approximated or silently
dropped.

### 5.3 Provenance IDs and analytical step IDs

Maintain two independent identifiers:

```text
trace_node_id
    full nested ProgramIR/AST provenance and debugging trace

semantic_step_id
    non-overlapping analytical partition used for robustness analysis
```

Illustrative semantic steps:

```text
S01 FUNCTION_INTERFACE
S02 INIT
S03 LOOP_HEADER
S04 CONDITION
S05 UPDATE
S06 RETURN
```

Nested trace nodes may overlap. Semantic-step character spans may not overlap.
Every analyzed token may be assigned to at most one `semantic_step_id`.

---

## 6. Renderer contract

Each deterministic renderer returns:

```json
{
  "text": "rendered representation",
  "trace_spans": [
    {"trace_node_id": "N1", "start_char": 0, "end_char": 18}
  ],
  "semantic_step_spans": [
    {"semantic_step_id": "S01", "start_char": 0, "end_char": 18}
  ]
}
```

Required renderers:

```text
render_procedural_prose(ir)
render_controlled_nl(ir)
render_pseudocode(ir)
render_python_opaque(original_python, scope_information)
```

Renderers may not use an LLM, external API, stochastic paraphrasing, or random
synonym selection. Renderer templates and their hashes must be frozen before any
real-model output is inspected.

The renderer grammar must document the full bundle of surface changes made at
each C1-C3 transition. This audit supports interpretation but does not turn the
transitions into feature-orthogonal effects.

---

## 7. Identifier-opacity intervention

The renamer must be scope-aware rather than a textual replacement.

Rename consistently within the relevant binding scope:

- function name;
- formal parameters;
- local variables;
- loop-bound variables.

Preserve:

- Python keywords;
- builtins used as builtins;
- attribute names following `.`;
- external/global API symbols;
- external keyword labels;
- literals and operators;
- import names unless a separately validated rule proves safe.

Opaque names must be deterministic, valid Python identifiers, non-colliding, and
free of obvious English semantic content. The mapping must be stored for every
trajectory.

### 7.1 Required validation chain

```text
parse canonical
→ scope analysis
→ rename bound identifiers
→ parse opaque
→ compile canonical
→ compile opaque
→ scope-aware binding checks
→ normalized AST equivalence
→ differential execution
```

### 7.2 Differential execution

Use deterministic, task-specific input cases. Execute canonical and opaque
functions in isolated processes with time and resource limits. Deep-copy mutable
inputs before each execution.

Compare at minimum:

- returned value, including nested container content;
- exception type;
- observable mutation of every supplied positional or keyword input.

If either implementation times out, fails to compile, produces a different
return, raises a different exception type, or leaves different observable input
state, the trajectory fails validation.

Differential execution supplements rather than replaces AST and binding checks.

---

## 8. Stimulus validation

Validation must fail hard unless all of the following hold:

- source SHA256 matches;
- exactly 100 tasks and 50 families;
- exactly 3 variants per task per condition;
- exactly 6 conditions;
- exactly 1,800 trajectories per model;
- no empty or missing representation;
- all 300 canonical Python programs parse and compile;
- all 300 opaque Python programs parse and compile;
- scope-aware bindings are valid;
- normalized canonical/opaque ASTs are equivalent;
- differential execution passes;
- ProgramIR represents every source construct;
- interface name and ordered parameters exist in all IR-derived conditions;
- all trace and semantic-step spans are in range;
- semantic-step spans are non-overlapping;
- every required ProgramIR component has provenance;
- frozen NL and canonical Python anchors remain byte-identical;
- semantic-step token coverage satisfies Section 12.

Any failed contrast caused by stimulus, semantic, execution, or mapping validation
is `NOT EVALUABLE`, not `NOT SUPPORTED`.

---

## 9. Human audit

Before real model runs, generate an audit package containing:

1. 20 fixed-seed randomly selected tasks, with all 6 conditions and all 3
   variants;
2. the smallest deterministic construct-coverage set needed to cover every
   supported rare ProgramIR or renderer template not already present in the random
   sample.

Coverage must explicitly include, when present:

- `while`;
- `break`;
- boolean expressions;
- dictionary construction or access;
- attribute/method access;
- nested control flow;
- function interface rendering.

One task may satisfy multiple coverage obligations. The audit report must list the
construct-to-task coverage matrix and any manual corrections. Stimuli may be
corrected only before hashes and preregistration are frozen.

---

## 10. Prompt and inference

Use the same raw format as Experiment 7:

```text
Task:
<TASK>

Solution:
<REFERENCE>
```

Do not use chat templates, representation labels, generation, sampling, or
instructions such as “answer in Python”.

Inference requirements:

```text
Transformers direct causal forward
BF16
CUDA
teacher forcing
full-sequence causal masking
no Ollama
```

Tokenize base prompt plus reference as one string with a fast tokenizer. Use
offset mappings and explicitly record prompt/reference BPE boundary fusion.

Exclude the first reference token from every trajectory metric, including
semantic-step-balanced metrics.

---

## 11. Metrics

### 11.1 Primary metric

```text
mean_entropy_top20
```

At each analyzed token position:

1. obtain the causal next-token distribution over usable vocabulary IDs;
2. select the top 20 probabilities;
3. renormalize within the top 20;
4. compute Shannon entropy;
5. average over reference tokens except the first.

### 11.2 Mandatory robustness metrics

```text
mean_entropy_valid_vocab
semantic_step_balanced_entropy_top20
```

Also record:

```text
mean_p1
median_p1
mean_top20_mass
mean_reference_probability
mean_reference_surprisal
median_reference_rank
fraction_p1_ge_090
fraction_p1_ge_099
token_count
analysis_token_count
```

Robustness metrics do not receive additional mandatory significance thresholds.
They are used for preregistered directional-consistency classification.

---

## 12. Semantic-step-balanced entropy

For each IR-derived trajectory:

1. map each eligible analyzed reference token to at most one
   `semantic_step_id`;
2. compute mean token entropy within each represented semantic step;
3. average those step means with equal weight.

Token assignment rules:

- whitespace-only tokens are not assigned;
- punctuation-only tokens are not assigned;
- these tokens remain in the ordinary primary entropy;
- a BPE token whose character span is completely inside one semantic-step span is
  assigned to that step;
- a BPE token overlapping more than one semantic-step boundary is excluded from
  the step-balanced metric and marked `boundary_overlap`;
- the first reference token is excluded before mapping;
- `nl_exp7_anchor` does not receive this analysis.

Record:

```text
step_mapped_token_fraction
boundary_overlap_token_fraction
unmapped_token_fraction
step_mapped_token_fraction_all_analyzed
```

For the validation threshold, the eligible denominator contains analyzed tokens
that are neither whitespace-only nor punctuation-only. Require:

```text
step_mapped_token_fraction >= 0.90
```

for every IR-derived condition before real runs. All-token coverage is reported
separately and has no acceptance threshold. This denominator definition resolves
the otherwise contradictory combination of excluding punctuation from steps while
requiring high semantic-token coverage.

---

## 13. Task-level contrasts

First average the 3 variants inside each task and condition.

### C1 — transition to controlled natural language

```text
C1 = H(procedural_prose) - H(controlled_nl)
H1: C1 > 0
```

Interpretation: the complete deterministic transition from procedural prose to
controlled NL lowers predictive entropy under fixed procedural semantics. The
transition changes a bundle of surface properties.

### C2 — transition to pseudocode

```text
C2 = H(controlled_nl) - H(pseudocode)
H1: C2 > 0
```

Interpretation: the complete deterministic transition from controlled NL to
pseudocode lowers predictive entropy. It is not a pure estimate of symbolism.

### C3 — transition to canonical Python

```text
C3 = H(pseudocode) - H(python_canonical)
H1: C3 > 0
```

Interpretation: the complete deterministic transition from pseudocode to Python
lowers predictive entropy. It does not separately identify syntax familiarity,
canonicality, tokenizer alignment, indentation, or corpus frequency.

### C4 — identifier-opacity intervention

```text
C4 = H(python_opaque) - H(python_canonical)
```

C4 has a two-sided difference test and a separate practical-equivalence analysis.
No direction is assumed in advance.

### Sanity contrast

```text
H(nl_exp7_anchor) - H(python_canonical)
```

This is a historical continuity check, not a new primary hypothesis.

---

## 14. Inferential units

Primary unit:

```text
task, N = 100
```

Variants, trajectories, and tokens are repeated measurements, not independent
inferential units.

Mandatory clustering robustness unit:

```text
family, N = 50
```

---

## 15. Primary statistical analysis

For C1-C3, use a one-sided paired sign-flip test. For C4, use a two-sided paired
sign-flip test.

```text
sign-flip resamples: 200,000
Monte Carlo correction: (extreme + 1) / (resamples + 1)
bootstrap resamples: 100,000
confidence level: 99%
fixed RNG seed: 20260916
```

For every contrast and model report:

- mean and median task delta;
- SD of task deltas;
- Cohen `dz`;
- task sign counts and fractions;
- 99% task bootstrap CI;
- Monte Carlo sign-flip p-value;
- family sign counts and fractions.

Apply Holm correction across C1-C4 within each model. The frozen threshold is:

```text
Holm-adjusted p < 0.01
```

Do not report the Monte Carlo floor as an exact mathematical p-value.

---

## 16. Family-cluster robustness

Do not replace the primary task-level analysis.

For each contrast and model:

1. compute the mean contrast for each of 50 families;
2. run a family-level sign-flip test on the 50 family means with 200,000
   resamples and the same +1 correction;
3. cluster-bootstrap 50 families with replacement;
4. whenever a family is selected, include both of its tasks;
5. compute 100,000 bootstrap means and a 99% family-cluster CI.

Report these results as mandatory dependence robustness. They do not create a new
multiple-testing acceptance family.

---

## 17. C4 practical equivalence

Freeze the following model-specific smallest effects of interest from the closed
Experiment 7 model-specific NL-minus-Python effects:

```text
Qwen/Qwen3-4B:                         δ = 0.0360463677
microsoft/Phi-3-mini-4k-instruct:      δ = 0.0659223095
mistralai/Mistral-7B-Instruct-v0.3:    δ = 0.0635291352
```

Each margin is 10% of that model's previously measured Experiment 7 mean
NL-minus-Python H20 effect. These margins are fixed before Experiment 8.

Practical equivalence for a model requires:

```text
99% task-bootstrap CI(C4) entirely inside [-δ_model, +δ_model]
```

The two-sided difference test remains separate. Classify C4 per model as:

| Difference test | 99% CI relative to equivalence bounds | Interpretation |
|---|---|---|
| significant | not entirely inside bounds | statistically different and potentially practically meaningful |
| significant | entirely inside bounds | statistically detectable but practically small |
| not significant | entirely inside bounds | evidence for practical equivalence |
| not significant | crosses either bound | inconclusive |

Never interpret a nonsignificant p-value by itself as equivalence.

Cross-model practical equivalence requires the model-specific 99% CI to lie
inside its own frozen bounds in all three models. A replicated nonzero C4 effect
requires significance in all models, 99% CIs excluding zero, a common direction,
and the frozen task/family sign thresholds in all models. Otherwise report the
heterogeneity without forcing a universal conclusion.

---

## 18. Cross-model acceptance criteria for C1-C3

A sequential representational intervention is fully cross-model replicated only
if all three models satisfy:

- preregistered direction;
- Holm-adjusted `p < 0.01`;
- 99% task bootstrap CI entirely above zero;
- at least 70/100 task contrasts above zero;
- at least 35/50 family means above zero.

These criteria concern the complete representation transition. They do not
identify a single surface feature.

---

## 19. Result classification

For each model and directional contrast C1-C3, assign one status:

### `ROBUST`

The primary preregistered criterion passes, and both mandatory robustness metrics
have the same contrast direction as the primary H20 metric:

```text
valid-vocab direction matches
AND semantic-step-balanced direction matches
```

### `METRIC-SENSITIVE`

The primary criterion passes, but at least one mandatory robustness metric
reverses direction.

### `NOT SUPPORTED`

The primary preregistered criterion fails.

### `NOT EVALUABLE`

Stimulus, semantic-step, scope, execution, or other mandatory validation fails.

For C4, report the same technical validity status, the two-sided difference
result, the four-way practical-equivalence class from Section 17, and directional
robustness only when a nonzero direction is inferred. Near-zero equivalence is not
forced into a directional-consistency rule.

Robustness metrics receive no additional required p-value gate.

---

## 20. Experiment size and order

Per model:

```text
100 tasks × 3 variants × 6 conditions = 1,800 trajectories
```

Across models:

```text
1,800 × 3 = 5,400 trajectories
```

Use a fixed shuffled trajectory order and store it before real runs. The runner
must support safe resume, trajectory-level checkpoints, partial-row repair,
immutable-config checks, and crash-safe writes. Never merge raw rows from
different models.

---

## 21. Project layout

Implementation, when separately authorized, must use:

```text
experiment8_causal_decomposition/
├─ DESIGN.md
├─ PREREGISTRATION.md
├─ tasks.json
├─ stimuli/
│  ├─ procedure_ir.json
│  ├─ rendered_stimuli.json
│  ├─ validation_report.json
│  ├─ differential_execution_report.json
│  └─ audit_sample.csv
├─ src/
│  ├─ build_ir.py
│  ├─ renderers.py
│  ├─ opaque_renamer.py
│  ├─ validate_stimuli.py
│  ├─ run_experiment8.py
│  ├─ analyze_model.py
│  └─ compare_models.py
├─ tests/
│  ├─ test_ir.py
│  ├─ test_renderers.py
│  ├─ test_opaque_python.py
│  ├─ test_differential_execution.py
│  ├─ test_semantic_step_mapping.py
│  └─ test_token_alignment.py
├─ debug/
│  └─ synthetic_tasks.json
├─ results/
│  ├─ qwen3/
│  ├─ phi3/
│  └─ mistral/
└─ analysis/
   ├─ qwen3_summary.json
   ├─ phi3_summary.json
   ├─ mistral_summary.json
   ├─ cross_model_summary.json
   ├─ task_contrasts.csv
   └─ family_contrasts.csv
```

This document does not authorize creation of that directory or any implementation
artifact.

---

## 22. Required metadata and outputs

Per model, record at minimum:

```text
model id and exact revision
resolved commit hash
architecture and dtype
torch / transformers / CUDA versions
GPU name and CUDA_VISIBLE_DEVICES
tasks / IR / stimuli / renderer hashes
analysis and order seeds
batch size and top_k
tokenizer class and vocabulary sizes
usable vocabulary size
boundary-crossing counts
semantic-step coverage statistics
trajectory count and timestamps
```

Required per-model files:

```text
config.json
trajectory_order.json
tokens.csv
trajectories.csv
analysis_summary.json
task_contrasts.csv
family_contrasts.csv
```

---

## 23. Pre-run implementation sequence

When implementation is separately authorized:

1. implement ProgramIR interface/body schema;
2. inventory and support all source constructs;
3. implement deterministic renderers and provenance;
4. implement the scope-aware opacity renamer;
5. implement AST, binding, compile, and differential validation;
6. implement semantic-step token mapping and coverage validation;
7. generate the human audit package;
8. create five synthetic debug tasks outside the benchmark;
9. run unit tests and synthetic model smoke tests only;
10. resolve exact model revisions and freeze every hash/config/order;
11. report validation, coverage, token-length distributions, and ambiguities;
12. stop for scientific review.

Do not inspect entropy patterns on the real benchmark before freeze. Do not run
the 100-task model experiments without a later explicit authorization.

---

## 24. Interpretation guardrails

- C1-C3 are sequential representation transitions, not single-feature effects.
- C4 is identifier opacity, not a pure identifier-semantics effect.
- Low entropy is not correctness, reasoning quality, or semantic understanding.
- A nonsignificant difference is not evidence of equivalence.
- Raw entropy values must not be used to rank models across tokenizers.
- Tokenization can be part of the representation mechanism and is not “corrected
  away”.
- Node/step balancing reduces unequal semantic-step weighting but does not remove
  tokenization.
- Findings apply to the frozen procedural benchmark, prompt, models, and
  representations unless separately replicated elsewhere.
- No condition, threshold, model, or interpretation rule may be added or changed
  after real results are inspected.

---

## 25. Freeze checklist

The design is ready to freeze only when:

- `DESIGN_v2.md`, `PREREGISTRATION.md`, and `CONSISTENCY_AUDIT.md` contain no
  unresolved definitional contradiction;
- the Qwen immutable revision is resolved;
- the source task hash is reverified;
- ProgramIR construct inventory is complete;
- renderer grammars and interface mappings are specified;
- opacity renaming and differential-test policies are specified;
- semantic-step eligibility and coverage formulas are implemented and tested on
  synthetic tasks;
- all statistical seeds, margins, thresholds, and classifications are immutable;
- a human scientific review explicitly approves the freeze.

Until then, the status remains **candidate for freeze**, not frozen.
