# Dry Bean feature-selection study

Code and archived results for *Comparative Feature Engineering for Dry Bean
Classification Using Genetic Algorithms and Neighborhood Components Analysis*.

## Revision status

Stage 3 has a [frozen comparison and analysis plan](docs/stage3-protocol.md).
`python scripts/run_stage3.py --output results/revision_runs/stage3-core` runs
20 paired seeds at 200 unique evaluations per search. NCA runs separately with
`--suite nca`. These are long-running experiments; resumable commands and honest
interpretation boundaries are documented in the protocol. Preparation is not
evidence that the full study has completed.

Stage 2 adds equal-unique-evaluation random/GA comparisons, mutation-only and
tournament-only ablations, MI/RFE fixed-size baselines and bounded PCA/NCA workers.
See [the stage-2 protocol](docs/stage2-protocol.md) before interpreting its outputs.

```sh
python src/stage2.py --output results/revision_runs/stage2-validation
# Resume completed units without rerunning them (same settings/code/environment):
python src/stage2.py --output results/revision_runs/stage2-validation --resume
```

Defaults are development checks: seeds 42/52, budget 40, population 10, 90 seconds
per unit and 30 seconds for NCA. Exit code 2 means at least one unit timed out or
failed; inspect `unit-statuses.json` and per-unit logs. Such units have missing
scores, not zeros. `--retry-failed --resume` retries failed units under identical
settings and preserves prior attempt logs. A different runtime limit requires a
new output directory. An abandoned `runner.lock` after a hard process crash must
only be removed after confirming no other runner is using the directory.

Outputs include verified per-unit predictions, split row IDs, selection traces,
per-class metrics, a manifest, raw `metrics.csv`, and a `summary.csv` that shows
the number of completed seeds even when a method has no scores. No significance
claim should be made from this validation run.

The `revision/evaluation-audit` branch corrects the evaluation foundation. It is
not a completed new algorithm or a submission-ready manuscript. See
[the audit and revision gates](docs/revision-audit.md).

`results/comparison/` contains the original paper's results, unchanged. The
original implementation is preserved at commit `ef0bfe83f70d3cb91b11298711d2f0fca202fc98`.
Do not combine archived scores with corrected runs. The old `improved_ga` label
is retained in machine-readable outputs for traceability; describe it in prose
as the **higher-mutation / larger-tournament GA configuration**, not a new algorithm.

## Setup and tests

Use Python 3.12 and an isolated virtual environment:

```sh
python -m venv .venv
# Activate .venv for your operating system.
python -m pip install -r requirements-revision.txt
python -m pytest -q
```

## Execution

Quick full-dataset pipeline check (NOT publication evidence):

```sh
python src/main.py --output results/revision_runs/pilot --seeds 42 --population 6 --generations 2 --skip-nca
```

Corrected replication of the original five-seed design, including NCA:

```sh
python src/main.py --output results/revision_runs/corrected-five-seed --seeds 42 52 62 72 82
```

The second command can be expensive. Benchmark NCA before scheduling large runs.
Use a fresh output directory for each run: existing directories are refused,
including partial runs. Resume is not implemented yet. Each run records the data
hash, source hashes, versions, settings, split indices and completion status.
Set BLAS thread limits consistently when comparing runtimes.

Exploratory statistics (creates a new directory; never rewrites input results):

```sh
python src/statistical_analysis.py --results-dir results/comparison --output results/revision_runs/statistics-audit
```

All pairwise tests share one Holm correction family across methods and classifiers.
Repeated holdouts overlap: these tests are exploratory, not evidence from multiple
independent datasets. A nonsignificant difference does not establish equivalence.

## Protocol boundary

Outer stratified 80/20 split -> training-only inner 3-fold GA selection -> refit
selected models on outer training -> outer test evaluation. Scaling for each
fitness evaluation is fitted inside each inner fold. GA fitness uses logistic
regression; transfer to RF/SVM is a separate question, not classifier-specific
optimization. No hyperparameter search is performed in this baseline replication.

`ga_best_cv_macro_f1` is now the unpenalized inner score;
`ga_best_penalized_fitness` records the optimized objective. The legacy field
`best_score` is still penalized fitness. Shared data loading/scaling is excluded
from timings; reported totals must not be called end-to-end deployment latency.
