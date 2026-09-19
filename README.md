# Dry Bean feature-selection study

Code and archived results for *Comparative Feature Engineering for Dry Bean
Classification Using Genetic Algorithms and Neighborhood Components Analysis*.

## Revision status

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
