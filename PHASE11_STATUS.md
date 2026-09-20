# PHASE11_STATUS

Updated: 2026-09-20

## Phase
Phase 11 — Search Behaviour and Exact Reference Extension

## Current work package
**Work package A: Trace and provenance audit — COMPLETE**

**Gate A: PASSED.** Ordered evaluation identity, mask identity, archived fitness, outer-split identity, deterministically reconstructed inner-fold identity, method identity, and final selected mask are recoverable for every analyzed search trace. Phase-11 checkpoint/frontier work may proceed without rerunning frozen Stage 3.

## Repository state
- Repository: `samantr/dry-bean-classification`
- Source branch: `revision/evaluation-audit`
- Verified source-branch head at Phase-11 start: `6933c6eae7ec5e53f379c85e79fcd26567ad3cd3`
- Working branch: `revision/search-behaviour-exact-reference`
- Frozen Stage-3 execution source commit: `afab0d45a359ba5de7b798820d488a3698771e25`
- Latest completed GitHub Actions provenance-evidence commit before this status update: `71d8f019191b8ef4ae2ca29c927808d296ddaa78`
- Root `AGENTS.md`: not present at the inspected source ref
- Repository code search for `AGENTS.md`: no result
- History policy: no force-push, no history rewrite, no silent experimental changes

## Authoritative scientific baseline
- Frozen Stage-3 experiment remains immutable evidence.
- Manuscript baseline is the latest revised LaTeX submission package supplied with the project handoff; historical R4 five-seed results must not enter the revised study.
- The Phase-11 instruction file mentions a `_v2` package name, while the project handoff identifies the uploaded `dry_bean_tubitak_revision_submission_package.zip` as the latest authoritative package. The handoff/package contents are used as the manuscript baseline; the filename discrepancy is documented rather than guessed away.
- Frozen outer seeds: 1001–1020.
- Search objective: mean inner 3-fold macro-F1 minus `0.001 * |S| / 16`.
- Primary search budget: 200 **unique nonempty subset evaluations** per seed-method pair.
- Outer test data must not participate in subset search, checkpoint choice, frontier construction, or exact-objective optimization.
- Exhaustive search, if later approved by Gate B, is an exact reference for the frozen objective and not a 200-evaluation competitor.

## Work package A audit checklist

### A1. Evidence inventory
- [x] Located Stage-3 search traces in `results/validation/stage3-final/checkpoints.tar.gz`.
- [x] Located manifests/completion records for 180 core units and 20 NCA units.
- [x] Located final selected masks and selected feature indices/names.
- [x] Located cached unique-evaluation histories and per-candidate fitness/fold scores.
- [x] Located archived outer train/test row indices and prediction arrays.
- [x] Reconstructed exact inner fold original-row identities from the verified dataset, outer-training order, seed, frozen source, and scikit-learn protocol; stored digests in `inner_fold_inventory.csv`.
- [x] Located Stage-3 analysis files and frozen source revision.
- [x] Located authoritative configuration/environment manifests.
- [x] Recorded hashes and unit-level provenance in the Phase-11 audit outputs.

### A2. Per seed-method provenance checks
For all 100 search units:
- [x] exactly 200 distinct nonempty masks;
- [x] deterministic evaluation order 1–200 recoverable;
- [x] same initial sequence of 20 masks across the five search methods within every seed;
- [x] same outer row indices across all methods within every seed;
- [x] deterministic inner folds recoverable for every seed;
- [x] fitness reproduces as mean inner macro-F1 minus `0.001 * |S| / 16`;
- [x] archived `cv_macro_f1` reproduces from the three fold scores;
- [x] scaling is fitted inside each inner CV training pipeline;
- [x] no outer-test value is used by `budgeted_search`;
- [x] final selected mask is traceable to the evaluated history and selected-feature mapping.

### A3. Claim-to-evidence mapping
- [x] Built `results/validation/phase11-audit/CLAIM_TO_EVIDENCE.md`.
- [x] Checked current manuscript numerical results against frozen Stage-3 summaries/analysis.
- [x] Distinguished inner CV macro-F1, penalized fitness, outer-test macro-F1, retained original inputs, transformed dimensions, search/representation time, and downstream fit time.
- [x] Verified the supplied revised package's performance/trade-off summary CSV values against frozen Stage-3 evidence.
- [x] Flagged two metric-attribution wording defects for later manuscript correction.

## Work package A evidence
- `results/validation/phase11-audit/TRACE_AUDIT_REPORT.md`
- `results/validation/phase11-audit/trace_audit_summary.json`
- `results/validation/phase11-audit/trace_inventory.csv`
- `results/validation/phase11-audit/inner_fold_inventory.csv`
- `results/validation/phase11-audit/source_hash_audit.csv`
- `results/validation/phase11-audit/CLAIM_TO_EVIDENCE.md`

Checkpoint archive SHA256:
`89d34ac8c590c1e52c2f1d93bc4a1061c421c91ea178d48ace1b8c06cb3a921f`

Frozen dataset SHA256:
`a9efa69741c6c5d95167c962a13da23a456f7e940402a5add12639790e01f714`

## Verified findings
- 200 archived Stage-3 result units are present: 180 core plus 20 NCA, all complete.
- All archived result SHA256 values match unit-status records.
- The 100 stochastic-search units each contain 200 unique nonzero masks and 600 inner model fits.
- Every seed uses an identical initial 20-mask sequence across random search and the four GA variants.
- Every method within a seed uses the identical archived outer split.
- The frozen source and dataset hashes match the Stage-3 manifests.
- Dataset profile from the frozen file: 13,611 rows, 16 input descriptors, 7 classes; class counts BARBUNYA 1322, BOMBAY 522, CALI 1630, DERMASON 3546, HOROZ 1928, SEKER 2027, SIRA 2636.
- No Stage-3 rerun is required for provenance or checkpoint reconstruction.

## Manuscript issues discovered by the audit
1. In the abstract, the values +0.00025 and +0.00057 are described as being “under the search objective.” These are actually paired **outer-test logistic-regression macro-F1** differences after selection, not inner CV objective/fitness differences.
2. The Discussion contains the same attribution risk when describing the GA-vs-random LR paired differences as being under the logistic-regression objective.

These are wording/provenance defects, not numerical-result failures. They must be corrected when the manuscript is revised after the new Phase-11 evidence is frozen.

## Commands/actions executed
1. Located and verified `revision/evaluation-audit`.
2. Confirmed `6933c6e...` as the exact Phase-11 source-branch head.
3. Created `revision/search-behaviour-exact-reference`.
4. Created this durable status file immediately.
5. Located the frozen Stage-3 source commit `afab0d45...` from the execution summary.
6. Added `scripts/phase11_trace_audit.py`.
7. Added `.github/workflows/phase11-trace-audit.yml`.
8. Ran the audit on GitHub Actions with the frozen dependency versions.
9. Archived the generated evidence in the repository.
10. Extended the audit to record the frozen dataset profile and reran it successfully.
11. Built the manuscript claim-to-evidence map.

## Missing or unresolved
- Gate B exhaustive-search feasibility has not yet been benchmarked.
- Phase-11 checkpoint, frontier, and exact-reference outputs do not yet exist.
- Current manuscript wording defects identified above are intentionally not edited until the new Phase-11 evidence is frozen.
- Journal formatting/policy issues from the handoff remain separate later-stage tasks.

## Next action
Begin **Work package B: Exhaustive-search feasibility benchmark** on GitHub Actions. First implement an exact enumerator that reuses the frozen `SubsetEvaluator` scoring path, validate it against archived Stage-3 masks/scores and bit ordering, then benchmark 500–1,000 uncached masks on one frozen split. Do **not** launch full 20-split enumeration until Gate B records exact-score reproduction, projected runtime/disk cost, and a resume strategy.
