# Phase 11 — Current Manuscript Claim-to-Evidence Map

Prepared: 2026-09-20  
Scope: latest revised LaTeX manuscript supplied with the Phase-11 handoff, before any Phase-11 search-behaviour extension.

## Evidence precedence

1. Frozen Stage-3 unit archive and manifests under `results/validation/stage3-final/`.
2. Frozen experiment source commit `afab0d45a359ba5de7b798820d488a3698771e25`.
3. Phase-11 provenance audit under `results/validation/phase11-audit/`.
4. The supplied revised manuscript package only as the location of claims being checked. Historical R4 five-seed results are excluded.

The archived Stage-3 evidence commit is `6933c6eae7ec5e53f379c85e79fcd26567ad3cd3`. The Phase-11 provenance audit passed with no failures or warnings and verified 200 archived result units, including 100 search units.

## Claim audit

| Manuscript location / claim | Frozen evidence | Audit result |
|---|---|---|
| Abstract / Dataset: 13,611 instances, 16 descriptors, 7 classes | `phase11-audit/trace_audit_summary.json -> dataset_profile` | **Verified**: 13,611 rows, 16 inputs, 7 classes. |
| Dataset section: BOMBAY 522; DERMASON 3,546 | Same dataset profile; dataset SHA256 matches Stage-3 manifest | **Verified**: complete class counts BARBUNYA 1322, BOMBAY 522, CALI 1630, DERMASON 3546, HOROZ 1928, SEKER 2027, SIRA 2636. |
| Seeds 1001–1020; stratified 80:20 outer split | `core/config.json`; frozen `src/stage2.py`; trace audit | **Verified** for all 20 seeds. Every archived unit's outer indices reproduce exactly from the frozen dataset and seed. |
| All methods within a seed use the same outer train/test rows | `trace_inventory.csv`; `trace_audit_summary.json` | **Verified** for all 20 seeds and all 10 methods. |
| Outer test is absent from subset selection | frozen `src/stage2.py` and `src/budgeted_search.py`; source hashes verified by `source_hash_audit.csv` | **Verified**: search receives only `X_train, y_train`; test data are used downstream after selection. |
| Inner search uses 3-fold stratified CV, with scaling inside each fold | frozen `src/budgeted_search.py`; `inner_fold_inventory.csv` | **Verified**. Inner row indices were not stored directly in Stage-3 result JSON, but are deterministically reconstructed from the verified outer-training order, seed, frozen source, and scikit-learn protocol. |
| Search objective `mean inner macro-F1 - 0.001*|S|/16` | every archived search-history record; `trace_inventory.csv` | **Verified** for all 20,000 search evaluations. Per-record mean fold score and penalized fitness reproduce within audit tolerance. |
| Search budget 200 unique nonempty subsets; 600 inner model fits per search unit | search histories; `trace_audit_summary.json` | **Verified** for all 100 seed-search-method units. |
| Same initial sequence of 20 masks across random search and all four GA configurations within a seed | `trace_inventory.csv` | **Verified** for every seed. |
| Final selected mask is an evaluated candidate and maps to the archived selected feature indices | per-unit `result.json` inside `checkpoints.tar.gz`; audit | **Verified** for all 100 search units. |
| Frozen completion: 180 core units + 20 NCA units | `execution-summary.json`, unit-status files | **Verified**: all 200 units are complete and all result SHA256 values match their unit-status records. |
| Logistic regression `max_iter=5000`; random forest 200 trees; SVM RBF kernel | frozen `src/baseline_models.py` at `afab0d45...` | **Verified**. |
| Search learner is logistic regression in a StandardScaler pipeline | frozen `src/budgeted_search.py` | **Verified**. |
| GA: population 20, crossover probability 0.8, individual mutation-event probability 0.1 | frozen `src/budgeted_search.py`; Stage-3 config | **Verified**. |
| GA-standard mutation/tournament = 0.1/3; mutation-only = 0.2/3; tournament-only = 0.1/5; both = 0.2/5 | `GA_SETTINGS` in frozen `src/budgeted_search.py` | **Verified**. |
| Empty-subset repair and unseen immigrant after 10 duplicate-only generations | frozen `src/budgeted_search.py` | **Verified**. |
| MI and RFE retain 8 inputs; PCA/NCA output 6 dimensions; PCA/NCA still require all 16 original inputs | frozen `src/stage2.py`; core/NCA summaries | **Verified**. NCA uses `max_iter=200`. |
| Table: baseline LR/RF/SVM macro-F1 = 0.9365/0.9365/0.9419 with SD 0.0042/0.0045/0.0038 | `core/summary.csv`; `core/analysis.json` | **Verified**. |
| Table: random-search LR/RF/SVM = 0.9362/0.9391/0.9422; mean inputs 9.15 | same | **Verified**. |
| Table: GA-standard LR/RF/SVM = 0.9364/0.9387/0.9419; inputs 8.95 | same | **Verified**. |
| Table: GA-mutation LR/RF/SVM = 0.9367/0.9390/0.9427; inputs 8.70 | same | **Verified**. |
| Table: GA-tournament LR/RF/SVM = 0.9363/0.9391/0.9419; inputs 8.35 | same | **Verified**. |
| Table: GA-both LR/RF/SVM = 0.9364/0.9388/0.9419; inputs 8.35 | same | **Verified**. |
| Table: MI-8 = 0.9220/0.9184/0.9207 | same | **Verified**. |
| Table: RFE-8 = 0.9172/0.9129/0.9154 | same | **Verified**. |
| Table: PCA-6 = 0.9348/0.9383/0.9419; requires 16 originals | same | **Verified**. |
| Table: NCA-6 = 0.9361/0.9383/0.9418; requires 16 originals | `nca/summary.csv`; `nca/analysis.json` | **Verified**. |
| Highest search-based SVM mean is GA-mutation 0.9427 with 8.70 inputs; random search is 0.9422 with 9.15 | core summaries | **Verified**. |
| RF full-input mean 0.9365; stochastic-search means 0.9387–0.9391 | core summaries | **Verified**. |
| Standard GA minus random search, outer LR macro-F1: +0.00025; 13/20 wins | `core/analysis.json -> paired_comparisons` | **Verified**: exact mean +0.0002545717; 13 wins, 7 losses. |
| GA-mutation minus random search, outer LR: +0.00057; 14/20 wins | same | **Verified**: +0.0005701791; 14 wins, 6 losses. |
| GA-tournament and GA-both minus random search, outer LR: +0.00008 and +0.00024 | same | **Verified**: +0.0000787746 and +0.0002401248. |
| GA-standard minus random search, outer RF/SVM: -0.00036/-0.00027 | same | **Verified**: -0.0003608257/-0.0002706458. |
| GA-mutation minus random search, outer RF/SVM: -0.00007/+0.00051; SVM wins/losses 10/10 | same | **Verified**: -0.0000667467/+0.0005083388; SVM 10 wins, 10 losses. |
| All GA-vs-random paired-difference ranges cross zero for LR, RF, and SVM | `core/analysis.json` | **Verified** for all 12 GA-vs-random classifier comparisons. |
| Mean stochastic retained-input range 8.35–9.15 | core summaries | **Verified**. |
| Original-input reductions 42.8%, 44.1%, 45.6%, 47.8%, 47.8% for random/standard/mutation/tournament/both | derived from frozen mean input counts | **Verified arithmetic**: `100*(16-k)/16`, rounded to one decimal. |
| Search/representation times about 31 s for random and GA | `core/analysis.json -> summaries -> representation_seconds` | **Verified**: means 30.64–31.63 s. |
| Baseline RF fit 4.353 s; stochastic RF fits 2.764–3.005 s | core analysis summaries | **Verified**. |
| NCA representation learning 207.02 s average | `nca/analysis.json` | **Verified**: 207.0232472 s. |
| PCA representation 0.005 s; NCA 207.02 s; both require 16 original descriptors | core/NCA summaries | **Verified**. |
| Mean Jaccard: random 0.512; GA-standard 0.524; mutation 0.526; tournament 0.534; both 0.541 | `core/analysis.json -> subset_stability` | **Verified**. |
| MI-8 Jaccard 1.000; RFE-8 0.873 | same | **Verified**. |
| Extent, Solidity, ShapeFactor4 are selected 20/20 for every stochastic method; Roundness 19/20 except mutation 20/20 | same feature-count records | **Verified**. |
| 16 variables imply 65,535 nonempty masks | arithmetic | **Verified**: `2^16 - 1 = 65,535`. |

## Provenance defects or wording mismatches found

### 1. Abstract metric attribution must be corrected

The abstract currently says:

> “Under the search objective, the standard genetic algorithm exceeded equal-budget random search by only 0.00025 mean paired macro-F1 ...”

The values 0.00025 and 0.00057 are **not inner search-objective differences**. They are paired differences in **outer-test logistic-regression macro-F1 after subset selection**. The numbers are correct, but the phrase “Under the search objective” assigns them to the wrong metric.

Required correction after the evidence freeze: use wording such as **“On the outer test splits with logistic regression...”** or **“For downstream logistic-regression macro-F1...”**.

### 2. Discussion contains the same attribution risk

The statements that the paired GA advantage “under its own logistic-regression objective” ranges from 0.00008 to 0.00057 similarly refer to outer-test LR macro-F1, not the archived inner CV objective/penalized fitness. The discussion should distinguish:
- inner `cv_macro_f1`;
- inner penalized `fitness`;
- outer-test logistic-regression macro-F1.

This distinction is important for Phase 11 because checkpoint and exact-reference analyses will operate on the **training-only inner objective**, while outer-test results remain post-selection assessment.

## Package consistency check

The supplied revised package's `stage3_performance_summary.csv` and `stage3_tradeoff_summary.csv` were inspected against the frozen repository summaries. Their reported rounded means, standard deviations, input counts, representation/search times, RF fitting times, and Jaccard values agree with the frozen Stage-3 evidence.

## Work-Package-A conclusion

The current revised study has sufficient trace provenance to begin budget-checkpoint and frontier analysis. Ordered mask identity, evaluation order, fitness, method, outer split, deterministic inner-fold definition, and final selection are recoverable for every analyzed search trace. No Stage-3 rerun is needed for provenance.

Gate A remains scientifically valid **provided the manuscript metric-attribution wording above is corrected before final submission**. The wording issue does not invalidate the frozen numerical evidence.
