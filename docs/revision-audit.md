# Revision audit and evidence gates

## Scope

Baseline: `ef0bfe83f70d3cb91b11298711d2f0fca202fc98` on master.
The revision preserves the original dataset, code in git history, and all
`results/comparison` files. No manuscript conclusions are replaced by pilot results.

## Confirmed issues and disposition

| Issue | Evidence | Action |
| --- | --- | --- |
| Inner-CV preprocessing leakage | Outer-training StandardScaler output was passed into inner CV | Pass raw features; StandardScaler is inside the CV pipeline |
| Penalized fitness labelled CV macro-F1 | Fitness subtracts 0.001 times selected fraction | Store unpenalized score and penalized fitness separately |
| Results overwritten by default | main.py always wrote results/comparison | Require a new explicit directory; refuse existing targets |
| Uncorrected selective pairwise interpretation | Original tests exported unadjusted p-values | Add Holm correction across all 18 available comparisons |
| Silent duplicate averaging | pivot_table aggregated duplicate experimental units | Reject duplicates and nonfinite/out-of-range scores |
| Weak provenance | No recorded versions, split indices, or data/code hashes | Add a run manifest and exact outer split indices |
| 'Improved GA' novelty overstated | Only mutation and tournament parameters differ | Retain internal legacy label; revise the scientific claim |
| Timing scope incomplete | Shared loading/scaling not timed | Label scope explicitly; full deployment timing remains pending |

This is inner-validation leakage, not evidence that the outer test set was used
to train the selector. Its numerical impact must be measured, not assumed.

## Verification performed

- Eight automated tests passed in the initial phase-1 test suite.
- Full Dry Bean data pilot completed with seed 42, population 6, generations 2,
  all three classifiers, both GA configurations, and no NCA.
- Pilot scores are execution checks only; both GAs selected the same 9 features.
- Archived five-seed statistics were recomputed without retraining. Across the
  18 pairwise comparisons, all Holm-adjusted p-values are 1.0.
- Original raw CSVs already reproduce the paper's headline rounded mean scores.

## Corrections to the initial revision proposal

1. A weighted accuracy/compactness/stability objective is a hypothesis, not an
   established novel contribution. Define stability operationally before coding:
   variation in CV scores is predictive variability, not feature-subset stability.
   A fixed candidate subset has no intrinsic resampling stability without a
   clearly defined selection procedure across training resamples.
2. Increasing seed count does not turn overlapping train/test splits into
   independent datasets. Freeze the inference plan before inspecting final results.
3. There are 65,535 nonempty subsets for 16 features. Exhaustive evaluation is
   finite, but 3-fold evaluation already costs 196,605 model fits per outer split.
   Repeated splits and hyperparameter tuning multiply this cost. Benchmark first;
   an exhaustive reference is optional, not automatically inexpensive.
4. A smaller subset does not prove cheaper image acquisition: the descriptors
   can share the same segmentation and geometric calculations. Actual savings
   require measurement or a justified feature-cost model.
5. Failure to reject a difference is not proof of equivalence or noninferiority.
6. Numerical replication can differ because the original dependency versions
   were not recorded. Keep code-change effects distinct from environment effects.

## Remaining work, in order

1. Complete the source-to-manuscript claims/reference audit and current journal
   policy verification. The uploaded template is a starting constraint, not proof
   that current requirements are unchanged.
2. Benchmark corrected original-budget GA and NCA; complete a corrected replication.
3. Freeze a research protocol: primary question, deployment claim, paired split
   strategy, method/parameter search space, compute budgets, metrics and inference.
4. Add equal-budget random search and simple selection baselines. Separate the
   mutation-only and tournament-only ablations. Assess transfer from LR fitness
   to RF/SVM explicitly. Add per-class results, predictions and checkpoint/resume.
5. Review related stability-aware/evolutionary methods before defining an extension.
   No invented algorithmic novelty or unmeasured resource benefit is acceptable.
6. Execute frozen runs; report failures and negative results as well as gains.
   Avoid selecting a method using final-test results from this audit pilot.
7. Rewrite claims from verified evidence, then build the journal LaTeX submission
   with separate editable PDF figures, title page and verified declarations.

No full revised experiment, stability-aware method, exhaustive search, or final
journal manuscript has been completed in this foundation checkpoint.
