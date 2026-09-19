# Stage-2 engineering validation

## What changed

The bounded runner implements the protocol in `stage2-protocol.md`. Random search
and four GA configurations receive identical unique-evaluation budgets and
training-only objective functions. Mutation-only and tournament-only variants
isolate the two parameter changes that were confounded in the original paper.
MI-8 and RFE-8 provide simple feature-selection references; PCA-6 and NCA-6 are
representation baselines requiring all 16 original inputs.

The new generational GA has documented nonempty initialization, repair and a
logged stagnation-immigrant rule. It must not be described as numerically identical
to the old fixed-generation implementation or presented as a novel optimizer.

## Automated verification

29 tests passed before the validation run. Coverage includes exact unique-budget
accounting, a known optimum in a small fully enumerated test space, shared initial
populations, determinism, fold-local scaling, cached-fit suppression, timeout
recording, duplicate/stale-result protection, lock protection, checksum validation,
strict resume identity, classifier predictions, and all fixed-size baselines.

## Observed run outcomes

- 20 requested units (2 seeds x 10 methods): 18 completed and 2 timed out.
- Both timed-out units were NCA, at the preset 30-second limit. This is not a
  measured NCA classification failure, nor proof it cannot finish with more time.
- Completed units produced 54 classifier result rows. Every random/GA search
  used exactly 40 distinct subsets (120 inner CV model fits).
- All methods used identical outer partitions within each seed, with no overlap
  between that seed's training and test rows.
- Resume was run on the real output directory. All completed result SHA-256
  values and file modification times remained unchanged, and every unit remained
  at attempt 1. Timeouts were not silently retried. The overall run state remains
  `incomplete`, accurately reflecting the missing NCA results.
- No warnings or error traces appeared in completed worker logs.

For illustration only, mean SVM macro-F1 was 0.937012 for the full-input baseline,
0.939500 for random search, 0.937299 for standard GA, and 0.939213 for the GA with
both parameter changes. These two-seed, small-budget observations do not establish
superiority. They show why omitting random search could give a misleading account
of the value of genetic search. Do not choose a final method or tune its parameters
using this pilot's outer-test differences.

The browsable metrics, summaries and statuses are in `results/validation/stage2/`.
`checkpoints.tar.gz` contains the complete raw run, including prediction arrays,
split row IDs, fold-score histories, logs, checksums and the source/environment
manifest. Its root folder is `stage2-validation`. A hard timeout intentionally
leaves no predictive score for the affected unit.

## Scientific boundary

This is a two-seed development run on previously examined data, with B=40 and
population=10. It is not the final experiment. No confidence intervals, equivalence
claims, statistical superiority or novel-method claims are inferred from it.
Runtime caps include worker startup, selection/transform and downstream models;
they are safeguards, not clean measurements of optimizer speed alone.

The first phase's original-budget results and archived NCA scores are not combined
with stage-2 scores. Any incomplete NCA unit remains missing. The next scientific
step is to freeze the larger comparison and its inference plan, then collect full
results under declared compute limits. Journal reformatting follows the evidence
and claim revision, not this engineering check.
