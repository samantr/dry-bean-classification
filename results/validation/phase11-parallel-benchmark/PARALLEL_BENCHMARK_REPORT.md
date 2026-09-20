# Phase 11 Work Package B — Process-level Parallel Benchmark

- Validation: **PASS**
- Workers: 4
- Masks rescored: 1000
- Maximum absolute difference versus serial scores: 0.000e+00
- Parallel wall time: 87.09 s
- Effective child CPU use: 394.8%
- Projected wall time per complete split at this measured rate: 1.59 h
- Aggregate runner-hours for 20 splits at this measured rate: 31.71 h

## Interpretation

The parallel path changes only execution scheduling. Each worker creates the same frozen SubsetEvaluator for the same outer-training data and seed. The benchmark re-scores the exact 1,000 masks used by the serial feasibility benchmark and compares every fold score, CV mean, and penalized fitness.

Full exhaustive enumeration remains blocked until Gate B is explicitly recorded in PHASE11_STATUS.md.

## Failures

- None.
