# Phase 11 Work Package B — Exhaustive-search Feasibility Benchmark

- Validation status: **PASS**
- Full exhaustive enumeration launched: **NO**
- Benchmark split seed: 1001
- Archived Stage-3 masks recomputed: 25
- Maximum archived CV macro-F1 absolute difference: 0.000e+00
- Maximum archived fitness absolute difference: 0.000e+00
- Benchmark uncached masks: 1000
- Benchmark wall time: 239.53 s
- Mean wall time per mask: 0.239530 s
- Projected serial time for one 65,535-mask split: 4.36 h
- Projected serial time for 20 splits: 87.21 h
- Estimated compact CSV size per split: 6.5 MiB
- Estimated compact CSV size for 20 splits: 129.2 MiB
- Partition plan: 16 atomic parts/split of at most 4096 masks

## Exact-score validation

The benchmark uses the same SubsetEvaluator implementation as frozen Stage 3. All sampled archived masks are recomputed on the same outer-training split and compared against archived fold scores, CV macro-F1, cardinality, and penalized fitness.

All 16 single-feature masks and the full 16-feature mask are also scored to validate bit ordering, mask-to-feature mapping, nonempty-mask handling, and finite objective values.

## Resume strategy

Full enumeration, if approved after Gate B, will be partitioned by deterministic contiguous mask ranges. Each partition is written to a temporary file, validated, and atomically renamed. Resume will skip only validated final partitions. A split will not be called exact unless the merged mask set is exactly 1..65,535 with no duplicates, failures, or nonfinite scores.

## Failures

- None.

## Gate B note

This report records the required serial feasibility evidence. It does not authorize or launch the full 20-split exhaustive computation. The next decision must consider the measured runtime, disk estimate, exact-score reproduction, and whether safe process-level parallelism should be benchmarked before committing substantial GitHub Actions compute.
