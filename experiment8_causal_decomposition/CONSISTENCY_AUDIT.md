# Experiment 8 Design v2 — Consistency Audit

**Audit date:** 2026-09-16  
**Canonical documents audited:** `DESIGN_v2.md`, `PREREGISTRATION.md`  
**Historical document:** `Experiment_8_Causal_Decomposition_Design.md`  
**Result:** zero unresolved definitional contradictions in the canonical documents

---

## 1. Scope and precedence

`Experiment_8_Causal_Decomposition_Design.md` is retained as historical design
v1. It is superseded and must not be used to implement or analyze Experiment 8.

Canonical precedence is:

1. `PREREGISTRATION.md` for frozen hypotheses, thresholds, margins, and decision
   rules;
2. `DESIGN_v2.md` for scientific rationale, data contracts, validation, and
   future project structure;
3. the historical v1 file only for change provenance.

No implementation file or real Experiment 8 result exists at the time of this
audit.

---

## 2. Conflict-resolution table

| Topic | Old wording or ambiguity | Conflict/risk | Resolution | Canonical locations updated |
|---|---|---|---|---|
| Overall experiment | “causal decomposition” of specific properties | Could imply orthogonal single-feature identification | Experiment remains causal at the complete-representation intervention level; C1-C3 are sequential representational intervention effects and change feature bundles | Design v2 §§1, 13, 18, 24; Prereg §§1, 11, 15, 18 |
| C1 | “procedural explicitness” | Prose-to-controlled transition also changes formatting, capitalization, punctuation, lexical templates, and tokenization | C1 is the transition `procedural_prose → controlled_nl`; no single-feature attribution | Design v2 §13; Prereg §11 |
| C2 | “symbolic structure” | Controlled-NL-to-pseudocode transition changes more than symbolism | C2 is the complete transition `controlled_nl → pseudocode`; symbolism is not isolated | Design v2 §13; Prereg §11 |
| C3 | “Python-specific effect” | Pseudocode-to-Python transition jointly changes syntax, canonicality, indentation, tokenization, and corpus familiarity | C3 is the complete transition `pseudocode → python_canonical`; submechanisms remain unidentified | Design v2 §13; Prereg §11 |
| Program interface | Examples omitted the `def` interface even though all 300 source references are full functions | Python could contain function name/parameters while other conditions did not | ProgramIR is divided into `interface` and `body`; all conditions preserve interface arity, order, and binding roles; opaque names are reversibly mapped to the canonical interface | Design v2 §§4-6, 8; Prereg §§4-5 |
| Interface analysis | Interface treated implicitly as an ordinary statement or omitted | Ambiguous semantic alignment and token weighting | Interface is structurally separate from computational body but receives its own non-computational `FUNCTION_INTERFACE` semantic step | Design v2 §5; Prereg §5 |
| C4 label | “identifier familiarity / semantics” | Opaque renaming also changes frequency, morphology, length, and tokenizer segmentation | Canonical label is **identifier-opacity intervention**; no pure-semantics claim | Design v2 §§4.6, 7, 13, 24; Prereg §§1, 6, 11, 18 |
| C4 renaming | Deterministic string-like mapping and normalized AST comparison | Could rename the wrong binding or break external names while still appearing structurally similar | Scope-aware rename of function, parameters, locals, and loop bindings; preserve keywords, builtins, attributes, globals/APIs, and external keyword labels | Design v2 §7; Prereg §6 |
| C4 semantic validation | Parse plus normalized AST equivalence | Does not establish contextual runtime equivalence or equal mutation behavior | Require compile, binding validation, normalized AST equivalence, and deterministic differential execution comparing return, exception type, and observable mutation | Design v2 §§7-8; Prereg §6 |
| C4 null interpretation | `C4 ≈ 0` inferred from lack of significance | Nonsignificance is not equivalence | Freeze model-specific margins and require the 99% task-bootstrap CI to lie entirely inside the relevant margin | Design v2 §17; Prereg §14 |
| Qwen equivalence margin | Rounded or unspecified margin | Post-hoc flexibility | Freeze `0.0360463677` | Design v2 §17; Prereg §14 |
| Phi-3 equivalence margin | Rounded or unspecified margin | Post-hoc flexibility | Freeze `0.0659223095` | Design v2 §17; Prereg §14 |
| Mistral equivalence margin | Rounded or unspecified margin | Post-hoc flexibility | Freeze `0.0635291352` | Design v2 §17; Prereg §14 |
| C4 outcomes | Significant/non-significant binary | Conflates statistical detectability and practical size | Four classes: meaningful candidate, detectable-small, equivalent, inconclusive | Design v2 §17; Prereg §14 |
| Node-balanced metric | `node_balanced_entropy_top20` using potentially nested IR nodes | Nested spans can overlap and double-count tokens | Canonical metric is `semantic_step_balanced_entropy_top20`; nested `trace_node_id` is separate from non-overlapping `semantic_step_id` | Design v2 §§5, 11-12; Prereg §§5, 9-10 |
| Token-to-step mapping | No deterministic boundary policy | BPE tokens and structural tokens could be assigned inconsistently | Full-inside token maps once; multi-step overlap is excluded and flagged; first token excluded everywhere | Design v2 §12; Prereg §10 |
| Punctuation/coverage | Punctuation and whitespace excluded from steps, while coverage was phrased as 90% of non-whitespace tokens | Punctuation is non-whitespace, producing an internally inconsistent denominator | Eligibility denominator excludes both whitespace-only and punctuation-only tokens; all-token coverage is reported separately | Design v2 §12; Prereg §10 |
| NL anchor balancing | Metric requirements did not explicitly exclude the non-IR anchor | Artificial alignment could introduce a new subjective mapping | `nl_exp7_anchor` has no semantic-step-balanced analysis and remains a sanity condition | Design v2 §§4.1, 12; Prereg §§4, 10 |
| Primary inferential unit | Task `N=100` | No contradiction, retained for continuity | Task remains primary; variants/tokens are repeated measures | Design v2 §§14-15; Prereg §§8, 12 |
| Family dependence | Family robustness was sign counting only | Task bootstrap ignores two-task clustering | Add family-level sign-flip and cluster bootstrap sampling 50 families while preserving both tasks | Design v2 §16; Prereg §13 |
| Human audit | 20 fixed random tasks only | Rare constructs, especially `break`, could be missed | Add the smallest deterministic construct-coverage set beyond the 20 random tasks | Design v2 §9; Prereg §17 |
| Robustness inference | Secondary metrics recorded without a decision role | A primary effect could be claimed despite a reversed full-vocabulary or step-balanced effect | Pre-register directional consistency and `ROBUST`/`METRIC-SENSITIVE` statuses without extra p-value gates | Design v2 §§11, 19; Prereg §§9, 16 |
| Technical failure | Analysis failure could be mixed with lack of scientific support | Invalid stimuli are not evidence against a hypothesis | Add `NOT EVALUABLE` distinct from `NOT SUPPORTED` | Design v2 §§8, 19; Prereg §§6, 16 |
| C4 versus directional status | General robustness rule could force a direction on a near-zero equivalent effect | Direction is unstable and irrelevant under equivalence | C4 receives separate difference/equivalence classification; directional robustness applies only to an inferred nonzero direction | Design v2 §19; Prereg §16 |
| Multiple testing | Possible additional p-value gates for robustness metrics | Would silently expand the confirmatory testing family | Holm applies only to C1-C4 primary tests within model; robustness metrics use direction only | Design v2 §§15, 19; Prereg §§12, 16 |
| Model revisions | “Pin existing commits” despite missing Qwen commit metadata | Qwen could not initially be reproduced immutably | The local snapshot used by Experiment 6 was resolved as `1cfa9a7208912126459214e8b04321603b3df60c`; all three commits are now frozen in `model_registry.json` | Design v2 §§3, 25; Prereg §§3, 19 |
| Implementation authorization | Design contained a handoff prompt that could be mistaken for permission to build/run | Risk of premature implementation or result inspection | Canonical documents explicitly prohibit implementation and real runs in the current phase | Design v2 §§21, 23, 25; Prereg §§1, 19 |

---

## 3. Final canonical definitions

### 3.1 Experiment type

Experiment 8 is a deterministic causal intervention on complete representations
under fixed procedural semantics. It is not a feature-orthogonal decomposition.

### 3.2 C1-C3

```text
C1 = H(procedural_prose) - H(controlled_nl)
C2 = H(controlled_nl) - H(pseudocode)
C3 = H(pseudocode) - H(python_canonical)
```

Each is a sequential representational intervention effect. Positive values are
the preregistered directions. No transition is assigned to one isolated surface
feature.

### 3.3 C4

```text
C4 = H(python_opaque) - H(python_canonical)
```

C4 is an identifier-opacity intervention with a two-sided difference test and a
separate model-specific equivalence decision.

### 3.4 ProgramIR

```text
ProgramIR.interface.function_name
ProgramIR.interface.parameters[]
ProgramIR.body
```

The interface arity, parameter order, and binding roles are preserved across all
IR-derived conditions. Canonical lexical names appear in prose, controlled NL,
pseudocode, and canonical Python; opaque lexical names carry a reversible mapping
to the same ProgramIR interface.

### 3.5 Provenance and balancing

```text
trace_node_id
    nested provenance only

semantic_step_id
    non-overlapping analytical partition

semantic_step_balanced_entropy_top20
    equal average of per-step mean top-20 entropy
```

### 3.6 Coverage

The 90% gate uses eligible analyzed tokens: tokens that are neither
whitespace-only nor punctuation-only. Boundary-overlap and unmapped eligible
fractions are reported. All analyzed-token coverage is also reported without a
threshold.

### 3.7 Inference hierarchy

```text
PRIMARY: task, N=100
ROBUSTNESS: family cluster, N=50
```

Primary task inference is retained. Family sign-flip and family-cluster bootstrap
are mandatory robustness analyses.

### 3.8 Equivalence

```text
Qwen δ    = 0.0360463677
Phi-3 δ   = 0.0659223095
Mistral δ = 0.0635291352
```

Practical equivalence requires the model-specific 99% task-bootstrap C4 CI to be
entirely inside its bounds.

### 3.9 Result statuses

```text
ROBUST
METRIC-SENSITIVE
NOT SUPPORTED
NOT EVALUABLE
```

Technical invalidity is never interpreted as hypothesis failure. C4 additionally
receives its four-way difference/equivalence interpretation.

---

## 4. Terminology audit

The canonical documents were reviewed for the following terms and concepts:

```text
causal decomposition
C1
C2
C3
C4
identifier semantics
identifier opacity
equivalence
node-balanced entropy
semantic-step-balanced entropy
ProgramIR interface
family robustness
acceptance criteria
result classification
```

Expected canonical usage:

- “causal” appears only with the level of intervention qualified;
- C1-C3 are consistently described as sequential representation transitions;
- “identifier semantics” appears only in explicit denials of a pure-semantics
  interpretation;
- `node_balanced_entropy_top20` is absent from canonical definitions;
- `semantic_step_balanced_entropy_top20` is the only canonical step-balanced
  metric name;
- ProgramIR interface fields are identical in design and preregistration;
- task and family roles are non-conflicting;
- C4 difference and equivalence are always separated;
- no robustness metric has an extra confirmatory p-value requirement.

No unresolved terminology contradiction remains.

---

## 5. Numerical consistency audit

| Quantity | Canonical value | Design v2 | Preregistration | Status |
|---|---:|---:|---:|---|
| Tasks | 100 | 100 | 100 | consistent |
| Families | 50 | 50 | 50 | consistent |
| Variants/condition/task | 3 | 3 | 3 | consistent |
| Conditions | 6 | 6 | 6 | consistent |
| Trajectories/model | 1,800 | 1,800 | 1,800 | consistent |
| Total trajectories | 5,400 | 5,400 | 5,400 | consistent |
| Sign-flip resamples | 200,000 | 200,000 | 200,000 | consistent |
| Bootstrap resamples | 100,000 | 100,000 | 100,000 | consistent |
| CI level | 99% | 99% | 99% | consistent |
| RNG seed | 20260916 | 20260916 | 20260916 | consistent |
| Adjusted alpha | 0.01 | 0.01 | 0.01 | consistent |
| Task sign threshold | 70/100 | 70/100 | 70/100 | consistent |
| Family sign threshold | 35/50 | 35/50 | 35/50 | consistent |
| Step coverage gate | 0.90 | 0.90 | 0.90 | consistent |
| Qwen C4 margin | 0.0360463677 | exact | exact | consistent |
| Phi-3 C4 margin | 0.0659223095 | exact | exact | consistent |
| Mistral C4 margin | 0.0635291352 | exact | exact | consistent |

---

## 6. Changed sections relative to design v1

The following substantive areas were rewritten or added:

1. scientific scope and claim strength;
2. ProgramIR interface/body split;
3. interface rendering in every IR-derived condition;
4. full construct-coverage expectations;
5. trace versus semantic-step identity;
6. identifier-opacity scope and validation;
7. differential execution requirements;
8. semantic-step token assignment and coverage;
9. C1-C3 names and interpretations;
10. C4 equivalence margins and four-way interpretation;
11. family-cluster bootstrap and sign-flip;
12. construct-stratified human audit;
13. preregistered result statuses;
14. model-revision freeze requirements;
15. explicit stop boundary before implementation and real runs.

---

## 7. Open questions and freeze readiness

### Unresolved scientific contradictions

None.

### Open implementation specifications

These are not contradictions, but they must be resolved and hashed during the
later authorized implementation phase before real runs:

- the complete deterministic wording of every renderer template;
- the complete task-specific differential-input fixture set;
- the exact Unicode/token rule used to identify punctuation-only tokenizer spans;
- the final construct-to-audit-task coverage set;
- the final human decision on the generated audit sample.

None may be chosen after inspecting real Experiment 8 entropy output.

### Ready to freeze?

The scientific definitions, hypotheses, margins, statistics, and interpretation
rules are internally ready for final human review. All model revisions are now
immutable. Operational freeze still requires approval of the generated audit
sample and the final implementation manifest.

The correct next state is:

```text
DESIGN v2 complete
PREREGISTRATION candidate complete
CONSISTENCY AUDIT complete
→ human review
→ authorize implementation/validation only
→ freeze implementation artifacts and hashes
→ scientific review
→ separately authorize real runs
```

The present task stops here.
