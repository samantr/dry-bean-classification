# Phase 11 Work Package G — Evidence Classification

Updated: 2026-09-20

## Decision

The Phase-11 evidence supports **Outcome 3 at the frozen training-only search objective**: the tested GA configurations, especially the tournament-only configuration, reduce exact-objective regret relative to equal-budget random search at moderate and full budgets.

This is **not** evidence that GA produces a large or uniformly better outer-test predictive gain. The manuscript must distinguish search-objective efficiency from downstream predictive performance.

## Evidence supporting the classification

At budget 200, mean exact-objective regret is:

- random search: 0.0006632061;
- GA standard: 0.0002305169;
- GA mutation-only: 0.0002324780;
- GA tournament-only: 0.0001450014;
- GA both: 0.0002238122.

At budget 200, exact-optimum attainment counts across 20 frozen splits are:

- random search: 0/20;
- GA standard: 4/20;
- GA mutation-only: 1/20;
- GA tournament-only: 6/20;
- GA both: 4/20.

All four GA variants are within 0.001 exact-objective regret on all 20 splits at budget 200, whereas random search is within 0.001 on 17/20 splits.

Paired against random search at budget 200:

- GA standard has lower regret on 20/20 splits;
- GA mutation-only has lower regret on 19/20 splits;
- GA tournament-only has lower regret on 20/20 splits;
- GA both has lower regret on 18/20 splits.

At budget 150, GA standard and GA tournament-only also have lower regret than random search on 20/20 splits.

Training-only frontier coverage is also consistently stronger for GA candidate sets. Mean pairwise dominance coverage of GA over random versus random over GA is:

- GA standard: 0.9550 vs 0.8318;
- GA mutation-only: 0.9703 vs 0.8625;
- GA tournament-only: 0.9648 vs 0.7918;
- GA both: 0.9745 vs 0.8473.

For the 0.001 descriptive inner-macro-F1 tolerance, the mean smallest retained cardinality is 7.95 for random search versus 7.05, 6.95, 6.60, and 6.65 for the four GA variants respectively.

## Outer-test interpretation

The search advantage should not be rewritten as a large predictive-performance advantage.

At budget 200, mean paired outer-test logistic-regression macro-F1 difference versus random search is:

- GA standard: +0.0002546;
- GA mutation-only: +0.0005702;
- GA tournament-only: +0.0000788;
- GA both: +0.0002401.

For Random Forest and SVM, the mean paired differences are mixed in sign across GA configurations. This supports a narrow conclusion: the GA variants search the frozen objective more effectively under the tested budgets, but the resulting downstream predictive differences are small and classifier-dependent.

The exact-objective optimum itself is also not a uniformly better outer predictor than the budget-200 selected subset. This is expected because the exact reference optimizes only the frozen inner logistic-regression objective, not outer-test performance.

## Manuscript framing to use

The revised paper should present the contribution as an empirical attribution study of search value:

1. Under an equal budget of 200 unique subset evaluations, GA and random search produce similar outer predictive performance.
2. Exact enumeration shows that the GA variants, particularly tournament-only selection, generally finish closer to the attainable optimum of the frozen training-only objective.
3. The advantage emerges in search efficiency/regret rather than as a large improvement in test accuracy.
4. The result is limited to this 16-feature Dry Bean problem, frozen logistic-regression search learner, and tested budgets.
5. Do not claim a new GA, general superiority of evolutionary search, equivalence, or improved deployment performance.

## Required manuscript corrections already identified

- The values previously described as +0.00025 and +0.00057 “under the search objective” are outer-test logistic-regression macro-F1 differences after selection. They must be relabeled.
- Exact-reference results must be clearly separated from equal-budget competitor results.
- The 20-evaluation checkpoint is a protocol sanity check because all methods share the same first 20 masks.
- Repeated holdouts are dependent and are descriptive split-sensitivity evidence, not 20 independent datasets.

## Status

Work packages A–G are scientifically complete. Manuscript revision may now begin against the latest revised LaTeX baseline, with all numerical claims mapped to frozen Phase-11 evidence.
