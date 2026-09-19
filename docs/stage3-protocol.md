# Stage 3: frozen empirical comparison

Frozen on 2026-09-19, before inspecting the stage-3 test results. This is a
prospective analysis plan within an already explored dataset, not a registered
study or independent external validation. Stage-2 seeds 42 and 52 and the old
paper's seeds have already been examined.

## Practical question

On this tabular bean morphology dataset, does genetic subset search offer a
useful accuracy/compactness/runtime tradeoff relative to uniform random search
at the same evaluation budget? The practical motivation is simpler downstream
classification. Actual imaging, segmentation, measurement cost and deployment
latency were not measured; fewer columns do not establish acquisition savings.

## Fixed design

- Seeds: every integer from 1001 through 1020, with no selection based on scores.
- Stratified 80/20 splits, three training-only CV folds, B=200 distinct subset
  evaluations, population=20. All other settings remain stage2_v1.
- Nine core methods: all inputs, random search, the four documented GA
  configurations, MI8, RFE8 and PCA6. LR, RF and SVM evaluated for every method.
- NCA6 is a separately executed supplementary comparator because its quadratic
  training cost requires a substantially different runtime allowance. Use the
  same 20 seeds and original algorithm parameters, with 3600 seconds per unit.
  Do not pool timings measured on different machines. Report missing runs.
- Core timeout: 600 seconds per unit. All methods are sequential with numerical
  threads limited to one. The pilot's 30-second NCA cap was not evidence that
  NCA could not converge. A separate seed-42 runtime probe uses a 600-second cap.
- No tuning of penalties, retained dimensions, classifier parameters or budgets
  using these outer test results. Changes require a dated protocol amendment.

## Outcomes and analysis

Primary outcome: outer-test macro-F1 of LR, the classifier used for the search
objective. Primary comparison: ga_standard minus random_search. Secondary
comparisons: each other GA minus random_search; ga_mutation_only, ga_tournament_only
and ga_both minus ga_standard. RF/SVM transfer results are secondary.

Report per-method completed/requested counts, mean and sample SD across splits;
for each planned comparison report paired count, mean/median difference, range
and wins/ties/losses (absolute difference <= 1e-12 is a tie). These are descriptive
split-sensitivity summaries, not independent-sample confidence intervals.
Do not run the old Friedman/Wilcoxon script on these runs. No significance,
equivalence, noninferiority or population-generalization claim follows from this
plan. Incomplete comparisons must show the exact included seeds. Do not rank an
unfinished method against a complete one without making the missingness clear.

Report selected input counts, representation/training/prediction times and
per-class performance. For actual subset selectors report feature-selection
frequencies and mean pairwise Jaccard overlap across completed seeds. This mixes
split and algorithm randomness, is descriptive, and is not a stability objective
or causal feature-importance measure. Do not present PCA/NCA component count as
original-input reduction. Preserve predictions and exact row partitions.

The unit-level traces must demonstrate 200 unique evaluations and 600 CV model
fits for each search; all methods for a seed must have identical outer row IDs.
Summaries require checksum-verified checkpoints. Missing units remain missing.

## Contribution boundary

The two GA parameter changes are ablations, not a new algorithm. A larger study
cannot itself resolve the reviewer's novelty objection. If genetic search does
not show a useful advantage, retain that result and frame the paper as a bounded
empirical evaluation. A methodological-innovation claim would need separate
literature review, a defensible new method and further validation. Evidence here
is limited to one dataset and cannot establish deployment or cross-domain impact.

## Execution

Install requirements-revision.txt using Python 3.12, then run:

```sh
python scripts/run_stage3.py --output results/revision_runs/stage3-core
python scripts/run_stage3.py --output results/revision_runs/stage3-core --resume
python scripts/run_stage3.py --suite nca --output results/revision_runs/stage3-nca
python scripts/analyze_stage3.py --input results/revision_runs/stage3-core --output results/revision_runs/stage3-core-analysis
```

The NCA suite may take up to 20 hours at its cap; the core suite can also take
hours. Run on a machine that can remain active, retaining the entire output
directory. A resumed run must use the same source, environment and configuration.
Failure retry requires --resume --retry-failed. Do not modify source mid-run.
These commands do not automatically publish results or claim study completion.
