# PHASE11_STATUS

Updated: 2026-09-20

## Phase
Phase 11 — Search Behaviour and Exact Reference Extension

## Repository state
- Repository: `samantr/dry-bean-classification`
- Source branch: `revision/evaluation-audit`
- Verified source-branch head at Phase-11 start: `6933c6eae7ec5e53f379c85e79fcd26567ad3cd3`
- Working branch: `revision/search-behaviour-exact-reference`
- Frozen Stage-3 execution source commit: `afab0d45a359ba5de7b798820d488a3698771e25`
- Frozen dataset SHA256: `a9efa69741c6c5d95167c962a13da23a456f7e940402a5add12639790e01f714`
- Frozen checkpoint archive SHA256: `89d34ac8c590c1e52c2f1d93bc4a1061c421c91ea178d48ace1b8c06cb3a921f`
- No force-push or history rewrite has been used.

## Scientific baseline
- Historical R4 five-seed results are superseded and must not be mixed into this study.
- Frozen outer seeds: 1001–1020.
- Search objective: mean inner 3-fold macro-F1 minus `0.001 * |S| / 16`.
- Equal search budget: 200 distinct nonempty subset evaluations per seed-method pair.
- The first 20 subset masks are identical across random search and all four GA configurations within each seed.
- Outer test data are excluded from subset search, checkpoint choice, frontier construction, and exact-objective optimization.
- Exact exhaustive enumeration is an attainable-objective reference, not a 200-evaluation competitor.

## Work-package status
- **A — Trace and provenance audit: COMPLETE / Gate A PASSED.**
  - All 100 stochastic search traces have 200 unique nonzero masks and recoverable evaluation order.
  - Outer splits, deterministic inner folds, frozen fitness, final masks, and source/data hashes are verified.
- **B — Exhaustive-search feasibility: COMPLETE / Gate B PASSED.**
  - Archived score reproduction max difference: `0.000e+00`.
  - 4-process exact benchmark reproduced serial scores exactly.
- **C/D — Checkpoints and training-only performance-cardinality frontier: COMPLETE / VALIDATED.**
  - GitHub Actions run `35514011863`: SUCCESS.
  - Checkpoints: 20, 50, 100, 150, 200 unique evaluations.
  - Budget-200 metrics/predictions reproduce frozen Stage 3.
- **E — Exact reference: COMPLETE / VALIDATED.**
  - GitHub Actions run `35513927921`: SUCCESS.
  - 20/20 outer-training splits × 65,535 nonempty masks = 1,310,700 subset objectives.
  - Total inner logistic-regression fits: 3,932,100.
  - Heavy per-mask archives remain as GitHub Actions artifacts; compact summaries/hashes are committed.
- **F — Regret and attainment: COMPLETE / VALIDATED.**
  - GitHub Actions run `35525796410`: SUCCESS; validation failures: none.
- **G — Interpretation gate: COMPLETE.**
  - Classification: Outcome 3 for the frozen search objective. GA configurations generally reduce exact-objective regret relative to equal-budget random search; downstream outer predictive differences remain small and classifier-dependent.

## Key frozen Phase-11 result
At budget 200, mean exact-objective regret is:
- random search: 0.0006632;
- GA standard: 0.0002305;
- GA mutation-only: 0.0002325;
- GA tournament-only: 0.0001450;
- GA both: 0.0002238.

Paired against random search at budget 200, lower regret occurs on:
- GA standard: 20/20 splits;
- GA mutation-only: 19/20;
- GA tournament-only: 20/20;
- GA both: 18/20.

Exact-optimum attainment at budget 200:
- random search: 0/20;
- GA standard: 4/20;
- GA mutation-only: 1/20;
- GA tournament-only: 6/20;
- GA both: 4/20.

All four GA variants are within 0.001 exact-objective regret on all 20 splits at budget 200; random search is within 0.001 on 17/20.

## Interpretation boundary
The paper may state that the tested GA configurations, particularly tournament-only selection in this experiment, approached the frozen training-only objective more efficiently than equal-budget random search. It must not convert this into a claim of a large, general, or classifier-independent predictive advantage. Repeated holdouts are dependent and are used for descriptive split-sensitivity analysis rather than independent-dataset inference.

## Manuscript revision status
- Latest revised LaTeX manuscript has been rewritten around frozen Phase-11 evidence.
- The earlier metric-attribution defect has been corrected: +0.00025 and +0.00057 are paired **outer-test logistic-regression macro-F1** differences, not search-objective differences.
- Current compiled anonymous main manuscript: 12 pages, 249-word abstract, 5 keywords, 4 tables, 3 figures, 13 references.
- Separate title page is compiled separately.
- Anonymous-main identity scan passed: no author names, affiliations, institutional name, identifying repository URL, or author PDF metadata.
- References use full journal names according to the current journal website.
- All PDFs were rebuilt, preflighted, rendered, and visually inspected after the final text/figure update.
- Figures were generated programmatically from frozen numeric CSVs; no generative-AI image creation/modification was used.
- Current figure PDFs embed Tinos CID TrueType; genuine Times New Roman is still required for strict compliance with the live figure-font instruction.

## Live journal-policy audit
Rechecked on 2026-09-20:
- journal operates double-blind review;
- initial research manuscripts are limited to 15 pages;
- research abstract limit is 300 words;
- research articles may have no more than 10 figures/tables combined;
- separate title page is required;
- current website says journal titles should not be abbreviated;
- current generative-AI policy requires full disclosure and author accountability and prohibits generative-AI image creation/modification.

Documented conflicts:
1. the live Information for Authors page contains a generic “MS Word only” sentence but later explicitly and repeatedly requires the journal template and LaTeX submission;
2. the supplied template contains older reference-abbreviation wording while the current website explicitly requires full journal names.

## Remaining blockers before submission
1. Final independent human scientific review, meaningful revision/adoption, and approval by both authors.
2. Re-export figures with genuine/licensed Times New Roman.
3. Confirm Editorial Manager upload slots/file types at submission time because of the Word-vs-LaTeX website contradiction.
4. Confirm double-blind handling of the public code/audit repository and any reviewer-facing supplemental files.
5. Ensure CRediT entries in the portal match the title-page statement and are approved by both authors.

No manuscript has been submitted and no journal/editor has been contacted.
