# PHASE11_STATUS

Updated: 2026-09-20

## Phase
Phase 11 — Search Behaviour and Exact Reference Extension

## Current work package
**Work package A: Trace and provenance audit**

Gate A is **NOT PASSED**. No convergence, Pareto, checkpoint, or exhaustive-search interpretation will begin until ordered trace provenance is independently recoverable.

## Repository state
- Repository: `samantr/dry-bean-classification`
- Source branch: `revision/evaluation-audit`
- Working branch: `revision/search-behaviour-exact-reference`
- Source commit: pending direct verification from the first Phase-11 commit parent
- Last-observed Stage-3 reference from the project handoff: `6933c6eae7ec5e53f379c85e79fcd26567ad3cd3` (reference only; not yet asserted as current source head)
- Root `AGENTS.md`: not present at the inspected source ref
- Repository code search for `AGENTS.md`: no result
- History policy: no force-push, no history rewrite, no silent experimental changes

## Authoritative scientific baseline
- Frozen Stage-3 experiment remains immutable evidence.
- Manuscript baseline is the latest revised LaTeX submission package; historical R4 five-seed results must not enter the revised study.
- Frozen outer seeds: 1001–1020.
- Search objective to verify: mean inner 3-fold macro-F1 minus `0.001 * |S| / 16`.
- Primary search budget: 200 **unique nonempty subset evaluations** per seed-method pair.
- Outer test data must not participate in subset search, checkpoint choice, frontier construction, or exact-objective optimization.
- Exhaustive search, if later approved by Gate B, is an exact reference for the frozen objective and not a 200-evaluation competitor.

## Work package A audit checklist

### A1. Evidence inventory
- [ ] Locate Stage-3 search traces.
- [ ] Locate manifests/completion records.
- [ ] Locate final selected masks.
- [ ] Locate cached fitness values.
- [ ] Locate outer split row indices.
- [ ] Locate inner fold definitions.
- [ ] Locate outer prediction files or verifiable prediction references.
- [ ] Locate Stage-3 analysis scripts.
- [ ] Locate authoritative configuration/environment manifests.
- [ ] Record file hashes and row counts for evidence used in Phase 11.

### A2. Per seed-method provenance checks
For every analyzed seed-method pair:
- [ ] exactly 200 distinct nonempty masks;
- [ ] deterministic evaluation order recoverable;
- [ ] same initial sequence of 20 masks across search methods within a seed;
- [ ] same outer row indices and inner folds;
- [ ] fitness equals mean inner macro-F1 minus `0.001 * |S| / 16`;
- [ ] scaling is fitted inside each inner training fold;
- [ ] no outer-test value is used during search;
- [ ] final selected mask is traceable to an evaluated candidate.

### A3. Claim-to-evidence mapping
- [ ] Build a table mapping every current manuscript numerical claim to frozen evidence.
- [ ] Add planned Phase-11 claims only after the required evidence exists.
- [ ] Explicitly distinguish inner fitness, inner macro-F1, outer-test metrics, retained original inputs, transformed dimensions, search time, and downstream fit time.

## Initial source precedence
1. Frozen Stage-3 repository outputs/manifests/protocol/code on the verified evaluation-audit lineage.
2. Latest revised LaTeX submission package for current manuscript wording and tables.
3. Phase-11 protocol/instructions for new analysis rules.
4. Historical R4 Word file only for recovering explanatory prose; never for revised numerical results.

## Commands/actions executed
1. Located source branch `revision/evaluation-audit`.
2. Checked root `AGENTS.md` at that ref: not found.
3. Searched repository for `AGENTS.md`: no result.
4. Created working branch `revision/search-behaviour-exact-reference` from `revision/evaluation-audit`.
5. Created this durable Phase-11 status file before new optimization work.

## Evidence findings
No scientific provenance finding is yet marked verified. The next action is repository-wide trace/manifests/configuration inventory on the working branch, followed by row-count/hash and per-unit audit.

## Missing or unresolved
- Exact source-head commit SHA still needs direct confirmation.
- Complete ordered Stage-3 trace coverage is not yet established.
- No claim-to-evidence table has yet been frozen.
- Gate A remains closed.
- Gate B exhaustive feasibility work has not started.

## Next action
Inventory all Stage-3 evidence paths and identify the files that jointly recover, for each seed-method unit, evaluation order, mask, fitness, split/fold identity, and final selection. Record gaps rather than rerunning Stage-3 for convenience.
