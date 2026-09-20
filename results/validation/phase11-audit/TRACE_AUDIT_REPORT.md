# Phase 11 Work Package A — Trace Provenance Audit

- Gate A: **PASS**
- Archived result units inspected: 200
- Search units inspected: 100
- Checkpoint archive SHA256: 89d34ac8c590c1e52c2f1d93bc4a1061c421c91ea178d48ace1b8c06cb3a921f
- Frozen source commit: afab0d45a359ba5de7b798820d488a3698771e25
- Dataset SHA256 match: True
- First 20 masks identical across all five search methods within every seed: True
- Outer split identical across all methods within every seed: True

## Important provenance note

Stage-3 result files archive per-candidate fold scores but not inner fold row indices. Exact inner folds are deterministically recoverable from the verified dataset, outer-training order, seed, frozen scikit-learn protocol, and source code. inner_fold_inventory.csv records reconstructed original-row digests for all 20 seeds.

## Failures

- None.

## Warnings

- None.

## Generated evidence

- trace_audit_summary.json
- trace_inventory.csv
- inner_fold_inventory.csv
- source_hash_audit.csv
