# Stage 2: budget-controlled comparisons (development protocol v1)

## Question and scope

Does genetic search outperform uniform random subset search at the same number
of unique LR cross-validation evaluations, and do mutation/tournament changes
help separately? Assess transfer to RF and SVM, prediction quality, selected
original-input count and measured runtime. This is an empirical comparison, not
a claim of a new algorithm, deployment savings, equivalence or superiority.

The protocol is fixed before the stage-2 validation run. Seeds 42 and 52 are
development checks, not fresh confirmatory evidence. The dataset has already
been used during method development. A final study requires a separately frozen
analysis plan; additional seeds alone do not erase that history.

## Paired design

- Same stratified 80/20 outer split and three inner stratified folds per seed.
- Each CV estimator is StandardScaler followed by LR (max_iter=5000); preprocessing
  fits inside each training fold. The selector receives only outer training data.
- Objective: mean inner macro-F1 minus 0.001 times selected fraction. Save both
  quantities, all fold scores and the complete evaluation trace.
- Random search draws uniformly from nonempty binary masks without replacement.
- Four GA configurations use per-gene mutation / tournament size (0.1,3), (0.2,3),
  (0.1,5), (0.2,5). Individual mutation=0.1 and two-point crossover=0.8 are fixed.
- All five searches start from the same initial subset sequence, use fresh caches,
  and spend exactly B unique subset evaluations (3B CV fits). Cache hits do not
  consume budget. Budget is not matched to wall-clock time.
- The budget-controlled generational GA differs from legacy fixed-generation
  eaSimple: initialization is unique and nonempty; empty offspring are repaired;
  after 10 consecutive duplicate-only generations, one unseen random immigrant
  is inserted. All such events are logged, not hidden. Final selection uses the
  best evaluated fitness, then fewer features, then smaller binary mask.
- Validation settings: B=40, population=10. Suggested later comparison setting:
  B=200, population=20, subject to protocol freeze and runtime validation.

## Non-search baselines

- All 16 original inputs.
- Mutual-information ranking, retain 8 inputs (fixed, not tuned on test data).
- RFE with fold-independent outer-training StandardScaler+LR, retain 8 inputs,
  step=1. This is a fixed-size baseline, not a CV-tuned wrapper.
- StandardScaler+PCA with 6 components.
- StandardScaler+NCA with 6 components and max_iter=200; separately bounded.

The non-search baselines do not have equal search budgets; report their actual
times separately. PCA/NCA require all original inputs despite smaller output
dimension. The 8/6 choices are fixed comparators, not asserted optima. No L1,
exhaustive search or stability-aware objective is introduced in this stage.

## Evaluation and runtime

Refit LR, RF (200 trees) and RBF-SVM on each resulting outer-training representation,
then evaluate once on the outer test split. Scale LR/SVM inputs using a pipeline;
RF uses the representation directly. Record predictions, true labels, row IDs,
per-class precision/recall/F1, confusion matrix, macro-F1, balanced accuracy and
accuracy. Store fit and predict timings separately from selection/representation
time. Do not call model-only times full image-to-decision deployment latency.

Every (seed, method) is a separate worker with a wall-clock timeout. A timeout or
error is an outcome, never a score of zero or an excuse to substitute an archived
result. Completed units are checkpointed atomically and verified by SHA-256 on
resume. Configuration, dataset, environment and source hashes must match. A
changed timeout also requires a new run directory. Retries of failed units require
an explicit flag; their previous status/log is retained. Resume is at unit level,
not halfway through a GA or model fit.

Run all workers sequentially with one numerical-library thread for this pilot.
Validation timeout limits: 90 seconds per non-NCA unit; 30 seconds per NCA unit.
These are engineering safeguards, not computational-performance endpoints for
a final scientific comparison.

## Interpretation

Do not compute significance claims from the two-seed validation. Summaries must
show completion counts and missing units. Never rank methods that did not finish
as though they had measured predictive performance. Repeated holdouts overlap;
future inference must account for the evaluation design and multiple comparisons.
Between-run feature frequencies or Jaccard overlap are descriptive stability
checks, not an optimized stability objective or causal feature importance.
