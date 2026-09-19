# Stage 3 progress — 2026-09-19

The fixed 20-seed, B=200 protocol, resumable launcher and checksum-verified
descriptive analysis are implemented. All 31 tests pass. The analysis was also
run on the complete stage-2 validation directory: its two missing NCA units
remain explicitly missing, and paired comparisons use only completed pairs.

## NCA runtime probe

Seed 42, the full dataset, six components, max_iter=200 and one numerical-library
thread completed within the new 600-second limit. Unit time was 157.64 seconds;
representation fitting/transformation was 151.61 seconds. This is one development
seed and a runtime feasibility check, not a final-study result or universal
runtime estimate. Successful worker completion is not proof of optimization
convergence; the worker log should also be inspected for warnings.

LR/RF/SVM macro-F1 were respectively 0.933016, 0.935211 and 0.935239. Do not use
these scores to choose final-study parameters. The full checkpoint, predictions,
split indices, status and environment/source manifest are preserved under
`results/validation/stage3-nca-runtime/`. The earlier 30-second NCA timeouts do not
establish methodological failure.

## Remaining execution and manuscript work

The core study was launched at `results/revision_runs/stage3-core` after the NCA
probe finished. Inspect its manifest and unit statuses for current progress;
this note does not assert completion. Resume using the command in the protocol.
The 20-seed supplementary NCA suite remains to be executed sequentially after
the core study to avoid competing workloads distorting timing measurements.

After all attempts finish, produce the planned descriptive analysis, audit
predictions and missingness, preserve the evidence, and rewrite the manuscript
using the supplied journal template. Neither a completed manuscript nor a new
methodological contribution is claimed at this stage.
