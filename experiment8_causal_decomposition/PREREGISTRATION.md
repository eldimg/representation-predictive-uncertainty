# Experiment 8 Preregistration

**Title:** Sequential Representational Interventions on Predictive Entropy  
**Version:** 2.0 candidate  
**Date:** 2026-09-16  
**Companion design:** `DESIGN_v2.md`  
**Status:** not frozen; no real Experiment 8 inference is authorized

---

## 1. Research question

When procedural semantics are fixed through a shared deterministic ProgramIR, how
do sequential transitions among complete representations change within-model
next-token predictive entropy?

C1-C3 estimate sequential representational intervention effects. They do not
estimate isolated effects of individual surface features. Each transition changes
a preregistered bundle of representational properties.

C4 estimates an identifier-opacity intervention. It does not isolate pure
identifier semantics.

---

## 2. Frozen source and sample

Source benchmark:

```text
100 tasks
50 two-task families
3 NL references per task
3 canonical Python FunctionDef programs per task
```

Required source SHA256:

```text
dcdbc196f7dbf394fbf043690c764c470dbbbf9ee333e344c1f62f320a72a164
```

No task or variant may be edited, omitted, or replaced.

---

## 3. Models

Exactly three models:

```text
Qwen/Qwen3-4B
microsoft/Phi-3-mini-4k-instruct
mistralai/Mistral-7B-Instruct-v0.3
```

Resolved commits already available:

```text
Qwen3-4B:   1cfa9a7208912126459214e8b04321603b3df60c
Phi-3 Mini: f39ac1d28e925b323eae81227eaba4464caced4e
Mistral:    c170c708c41dac9275d15a8fff4eca08d52bab71
```

The Qwen revision was resolved from the locally cached snapshot used by the
closed Experiment 6 run and is recorded in `model_registry.json`. No floating
model revision is permitted in a real run.

---

## 4. Conditions

Exactly six conditions:

1. `nl_exp7_anchor`
2. `procedural_prose`
3. `controlled_nl`
4. `pseudocode`
5. `python_canonical`
6. `python_opaque`

`nl_exp7_anchor` and `python_canonical` remain byte-identical to the closed
benchmark. The other structured conditions are deterministic transformations
linked to the same ProgramIR.

Every IR-derived representation must include the same semantic procedure
interface:

```text
ProgramIR.interface.function_name
ProgramIR.interface.parameters[]
```

and the same represented body semantics. The first four IR-derived conditions
preserve canonical interface names. `python_opaque` preserves arity, parameter
order, and binding roles while deterministically renaming those identifiers and
recording a reversible mapping.

---

## 5. ProgramIR and rendering commitments

ProgramIR is divided into:

```text
interface
    function_name
    ordered parameters

body
    computational structure and semantic steps
```

Maintain separately:

```text
trace_node_id       nested provenance/debug mapping
semantic_step_id    non-overlapping analytical partition
```

No LLM, external service, stochastic paraphrase, or random synonym selection may
generate study stimuli.

The renderer implementation must fail on unsupported syntax. Silent
approximation and task dropping are prohibited.

---

## 6. Identifier-opacity intervention

The scope-aware renamer changes:

- the function name;
- formal parameters;
- local variables;
- loop-bound variables.

It preserves:

- keywords;
- builtins used as builtins;
- attribute names;
- external/global API symbols;
- external keyword labels;
- literals and operators.

Every canonical/opaque pair must pass:

```text
parse
scope analysis
rename
parse opaque
compile both
scope-aware binding validation
normalized AST equivalence
differential execution
```

Differential execution uses deterministic inputs and compares returned value,
exception type, and observable mutation of supplied inputs.

Failure makes the affected contrast `NOT EVALUABLE`.

---

## 7. Prompt and inference

Raw prompt:

```text
Task:
<TASK>

Solution:
<REFERENCE>
```

Inference:

```text
Transformers direct causal forward
BF16
CUDA
teacher forcing
full prompt+reference tokenization
fast-tokenizer offsets
no chat template
no generation
no sampling
no Ollama
```

The first reference token is excluded from every trajectory-level analysis.

---

## 8. Experiment size

```text
100 tasks × 3 variants × 6 conditions = 1,800 trajectories/model
1,800 trajectories × 3 models = 5,400 total trajectories
```

Primary inferential unit: task, `N = 100`.  
Family-cluster robustness unit: family, `N = 50`.

Tokens, trajectories, and variants are not independent inferential observations.

---

## 9. Metrics

Primary:

```text
mean_entropy_top20
```

Mandatory robustness:

```text
mean_entropy_valid_vocab
semantic_step_balanced_entropy_top20
```

Supportive descriptive metrics include `mean_p1`, top-20 mass, reference
surprisal, reference rank, token counts, and semantic-step mapping coverage.

Robustness metrics receive no separate mandatory significance threshold. Their
preregistered role is directional-consistency classification.

---

## 10. Semantic-step token mapping

The semantic-step metric is computed only for IR-derived conditions.
`nl_exp7_anchor` is excluded.

Rules:

- each analyzed token belongs to at most one `semantic_step_id`;
- whitespace-only and punctuation-only tokens remain in primary entropy but are
  excluded from semantic-step balancing;
- a BPE token fully contained in one step span is assigned to that step;
- a BPE token crossing step boundaries is excluded and recorded as
  `boundary_overlap`;
- the first reference token is excluded before mapping.

Required diagnostics:

```text
step_mapped_token_fraction
boundary_overlap_token_fraction
unmapped_token_fraction
step_mapped_token_fraction_all_analyzed
```

The validation denominator for `step_mapped_token_fraction` consists of analyzed
tokens that are neither whitespace-only nor punctuation-only. Each IR-derived
condition must achieve at least 0.90 coverage before real runs.

---

## 11. Primary contrasts and hypotheses

For each task, average the three variants inside each condition before computing
contrasts.

### C1

```text
C1 = H(procedural_prose) - H(controlled_nl)
H1: C1 > 0
```

Sequential transition effect; not a pure procedural-explicitness coefficient.

### C2

```text
C2 = H(controlled_nl) - H(pseudocode)
H1: C2 > 0
```

Sequential transition effect; not a pure symbolism coefficient.

### C3

```text
C3 = H(pseudocode) - H(python_canonical)
H1: C3 > 0
```

Sequential transition effect; not a pure Python-syntax or training-familiarity
coefficient.

### C4

```text
C4 = H(python_opaque) - H(python_canonical)
```

Two-sided identifier-opacity difference test plus a separate practical-
equivalence analysis.

### Sanity contrast

```text
H(nl_exp7_anchor) - H(python_canonical)
```

Historical pipeline-continuity check only.

---

## 12. Primary statistical tests

```text
C1-C3: one-sided paired task-level sign-flip
C4:    two-sided paired task-level sign-flip
resamples: 200,000
p correction: (extreme + 1) / (200,000 + 1)
bootstrap resamples: 100,000
bootstrap confidence: 99%
RNG seed: 20260916
```

Apply Holm correction across C1-C4 separately within each model. Required
adjusted threshold:

```text
adjusted p < 0.01
```

Report task mean, median, SD, Cohen `dz`, 99% bootstrap CI, and task/family sign
counts.

---

## 13. Family-cluster robustness

For every model and contrast:

- average the two task contrasts within each family;
- run a 200,000-resample sign-flip test on 50 family means;
- cluster-bootstrap families with replacement;
- preserve both tasks whenever a family is sampled;
- use 100,000 bootstrap resamples and report a 99% cluster CI.

This analysis is mandatory robustness and does not replace task-level inference.

---

## 14. C4 equivalence margins

Frozen before Experiment 8 from 10% of each model's closed Experiment 7 mean
NL-minus-Python H20 effect:

```text
Qwen/Qwen3-4B:                         δ = 0.0360463677
microsoft/Phi-3-mini-4k-instruct:      δ = 0.0659223095
mistralai/Mistral-7B-Instruct-v0.3:    δ = 0.0635291352
```

Evidence for practical equivalence requires:

```text
99% task-bootstrap CI(C4) ⊂ [-δ_model, +δ_model]
```

The difference test and equivalence decision are separate:

1. significant and CI not entirely within bounds: statistically different and
   potentially practically meaningful;
2. significant and CI entirely within bounds: statistically detectable but
   practically small;
3. not significant and CI entirely within bounds: evidence for practical
   equivalence;
4. not significant and CI crosses a bound: inconclusive.

A nonsignificant p-value alone is never evidence for equivalence.

Cross-model equivalence requires the model-specific 99% C4 CI to fall inside its
own frozen margin in all three models.

---

## 15. Directional cross-model replication rule

C1, C2, or C3 is fully cross-model replicated only if every model has:

- the preregistered positive direction;
- Holm-adjusted `p < 0.01`;
- 99% task CI wholly above zero;
- at least 70/100 positive task contrasts;
- at least 35/50 positive family means.

A nonzero C4 effect is cross-model replicated only if all models have significant
two-sided tests, 99% CIs excluding zero, the same direction, and the same frozen
task/family sign thresholds in that direction.

---

## 16. Result statuses

For directional C1-C3 results:

### `ROBUST`

Primary criteria pass, valid-vocabulary direction matches, and semantic-step-
balanced direction matches.

### `METRIC-SENSITIVE`

Primary criteria pass, but at least one mandatory robustness metric reverses
direction.

### `NOT SUPPORTED`

Primary preregistered criteria fail.

### `NOT EVALUABLE`

Mandatory stimulus, semantic-step, scope, execution, or pipeline validation
fails.

C4 additionally receives one of the four difference/equivalence interpretations
in Section 14. Directional robustness applies to C4 only when a nonzero direction
is inferred.

---

## 17. Human audit and pre-run validation

Before any real inference, inspect:

- 20 fixed-seed random tasks across all conditions and variants;
- the smallest additional deterministic set covering every supported rare
  construct or renderer template.

Coverage must explicitly include, if present, `while`, `break`, boolean
expressions, dictionaries, attribute/method access, nested control flow, and the
function interface.

All 300 canonical/opaque pairs must pass parse, compile, binding, AST, and
differential-execution validation. Every IR-derived condition must pass the 90%
semantic-step coverage gate.

Any failure blocks real runs.

---

## 18. Prohibited researcher degrees of freedom

After freeze, do not:

- add or remove conditions, tasks, variants, models, metrics, or contrasts;
- change directional hypotheses;
- change equivalence margins, thresholds, seeds, or correction families;
- redefine semantic-step spans or coverage after viewing real entropy results;
- substitute a model revision without recording a protocol deviation;
- interpret C1-C3 as isolated feature-level effects;
- interpret C4 as pure identifier semantics;
- interpret nonsignificance as equivalence;
- rank models using raw cross-tokenizer entropy.

Unexpected findings and protocol deviations must be labeled exploratory.

---

## 19. Freeze blockers

This preregistration is not yet frozen. Remaining blockers are:

1. complete human review of the canonical definitions and generated audit sample;
2. approve the technically validated renderer grammar, construct coverage,
   semantic-step mapping, and synthetic smoke result;
3. freeze the generated implementation manifest before any real run.

The current task stops at documentation. No Experiment 8 code, stimuli, model
runner, or real result has been produced.
