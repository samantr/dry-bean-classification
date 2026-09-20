# Phase 11 Work Packages C/D — Checkpoints and Frontiers

- Validation: **PASS**
- Seeds: 1001–1020
- Search methods: random search plus four frozen GA configurations
- Checkpoints: 20, 50, 100, 150, 200 unique evaluations
- Budget-20 role: protocol sanity check only; all methods share the first 20 masks within a seed.
- Selection quantities: training-only frozen fitness and inner macro-F1.
- Outer results: post-selection assessment only; never used to select checkpoint masks.
- Frontier objectives: maximize inner macro-F1 and minimize original-feature cardinality.
- Fixed descriptive inner-F1 tolerances: 0.001, 0.005.
- Hypervolume: not used.
- Per-evaluation timestamps: unavailable/trustworthy cumulative search time therefore not reconstructed.
- Budget-200 outer predictions and metrics are required to reproduce archived Stage-3 values exactly.

## Validation failures

- None.
