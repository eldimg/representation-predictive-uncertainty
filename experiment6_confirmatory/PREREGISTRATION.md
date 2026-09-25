# Experiment 6 Confirmatory — Frozen Analysis Plan

## Research question

For the same procedural task, does a teacher-forced Python solution trajectory exhibit lower next-token predictive entropy than a semantically matched natural-language solution trajectory in Qwen3-4B?

## Confirmatory design

- 100 frozen tasks.
- 50 task families, exactly 2 tasks per family.
- 3 frozen natural-language realizations per task.
- 3 frozen Python realizations per task.
- 600 teacher-forced trajectories total.
- Representation is never named in the model prompt.
- For every trajectory the model sees only:
  `Task: <same task description>\n\nSolution:\n`
  followed by the teacher-forced prefix.
- Temperature = 1.0.
- top_logprobs = 20.
- The first predicted token is excluded from the primary trajectory summary because the context before that token is identical across representations.

## Independence and aggregation

Tokens are NOT treated as independent observations.

Reference variants are NOT treated as independent confirmatory observations.

The primary inferential unit is the task (N = 100):

1. Compute mean token entropy for each trajectory.
2. Average the three NL trajectory means within each task.
3. Average the three Python trajectory means within each task.
4. Define task delta as `H_NL - H_Python`.
5. Perform inference across the 100 task deltas.

A secondary robustness analysis aggregates the two tasks within each family, producing 50 family-level deltas.

## Primary metric

`mean_entropy_top20`:
top-20 next-token entropy after renormalizing the returned top-20 probability mass.

The runner additionally records:
- top-20 probability mass;
- entropy lower and upper bounds accounting for omitted vocabulary mass;
- mean top-1 probability;
- fractions of positions with top-1 probability >= .90 and >= .99;
- reference-token visibility/rank when present in top-20;
- special-token events.

## Special-token handling

An Ollama event with `eval_count == 1`, empty textual response, and empty logprobs is recorded as a special-token event.

That token position is omitted from entropy aggregation but is not hidden.

A trajectory is pre-defined as invalid if more than 10% of its token positions are special-token events.

A task enters the primary analysis only when all 3 NL and all 3 Python trajectories are valid.

## Primary statistical test

One-sided paired sign-flip/permutation test on the 100 task deltas:

H1: mean(H_NL - H_Python) > 0

- alpha = 0.01
- 200,000 Monte Carlo sign flips
- fixed random seed = 20260916

## Secondary analyses

- 99% bootstrap confidence interval for mean task delta.
- One-sided Wilcoxon signed-rank test.
- One-sided paired t-test.
- Paired Cohen's dz.
- Number and fraction of tasks with positive delta.
- Same direction analysis at the 50-family level.
- Conservative entropy-bound comparison: NL lower bound versus Python upper bound.

## Pre-defined evidence criteria

The confirmatory pattern is considered robust only if all of the following hold:

1. Primary one-sided sign-flip p < .01.
2. The 99% bootstrap CI for mean task delta is entirely above zero.
3. At least 70 of 100 eligible tasks have `H_NL > H_Python`.
4. At least 35 of 50 eligible task families have positive mean delta.

These criteria are frozen before observing confirmatory results.

## Important scope

A positive result would support a model- and stimulus-specific statement:
for these frozen procedural tasks in Qwen3-4B, Python teacher-forced trajectories are more predictively concentrated than matched English natural-language trajectories.

It would NOT by itself establish a universal property of programming languages, natural languages, all LLMs, or human language.
