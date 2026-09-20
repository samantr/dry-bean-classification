#!/usr/bin/env python3
"""Phase 11 Work Package B: safe process-level parallel benchmark for exact scoring."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import resource
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from budgeted_search import SubsetEvaluator  # noqa: E402


_WORKER_EVALUATOR = None


def init_worker(X_train, y_train, seed):
    global _WORKER_EVALUATOR
    _WORKER_EVALUATOR = SubsetEvaluator(X_train, y_train, seed, penalty=0.001)


def score_mask(mask):
    record = _WORKER_EVALUATOR.evaluate(int(mask))
    return {
        "mask": int(record["mask"]),
        "selected_count": int(record["selected_count"]),
        "cv_macro_f1": float(record["cv_macro_f1"]),
        "fitness": float(record["fitness"]),
        "fold1": float(record["fold_scores"][0]),
        "fold2": float(record["fold_scores"][1]),
        "fold3": float(record["fold_scores"][2]),
    }


def load_outer_training(dataset: Path, seed: int):
    frame = pd.read_excel(dataset)
    X = frame.drop(columns="Class").to_numpy(dtype=float)
    y = LabelEncoder().fit_transform(frame["Class"])
    train, _ = train_test_split(
        np.arange(len(y)),
        test_size=0.2,
        stratify=y,
        random_state=seed,
    )
    return X[train], y[train]


def read_serial_sample(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    result = []
    for row in rows:
        result.append(
            {
                "mask": int(row["mask"]),
                "selected_count": int(row["selected_count"]),
                "cv_macro_f1": float(row["cv_macro_f1"]),
                "fitness": float(row["fitness"]),
                "fold1": float(row["fold1"]),
                "fold2": float(row["fold2"]),
                "fold3": float(row["fold3"]),
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--serial-sample", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--atol", type=float, default=1e-12)
    args = parser.parse_args()

    if args.workers < 2:
        parser.error("--workers must be at least 2")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    serial_rows = read_serial_sample(args.serial_sample)
    masks = [r["mask"] for r in serial_rows]
    if len(masks) < 500:
        raise ValueError("serial benchmark sample must contain at least 500 masks")
    if len(set(masks)) != len(masks):
        raise ValueError("serial benchmark sample contains duplicate masks")

    X_train, y_train = load_outer_training(args.dataset, args.seed)

    before_children = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=init_worker,
            initargs=(X_train, y_train, args.seed),
        ) as pool:
            parallel_rows = list(pool.map(score_mask, masks, chunksize=8))
    wall_seconds = time.perf_counter() - started
    after_children = resource.getrusage(resource.RUSAGE_CHILDREN)
    child_cpu_seconds = (
        after_children.ru_utime
        - before_children.ru_utime
        + after_children.ru_stime
        - before_children.ru_stime
    )

    by_mask = {r["mask"]: r for r in parallel_rows}
    failures = []
    comparison = []
    max_diff = 0.0
    for serial in serial_rows:
        parallel = by_mask.get(serial["mask"])
        if parallel is None:
            failures.append(f"missing parallel result for mask {serial['mask']}")
            continue
        fields = ("cv_macro_f1", "fitness", "fold1", "fold2", "fold3")
        diffs = [abs(serial[k] - parallel[k]) for k in fields]
        local_max = max(diffs)
        max_diff = max(max_diff, local_max)
        match = (
            serial["selected_count"] == parallel["selected_count"]
            and all(math.isclose(serial[k], parallel[k], rel_tol=0.0, abs_tol=args.atol) for k in fields)
        )
        if not match:
            failures.append(f"parallel score mismatch for mask {serial['mask']}")
        comparison.append(
            {
                "mask": serial["mask"],
                "selected_count": serial["selected_count"],
                "serial_fitness": serial["fitness"],
                "parallel_fitness": parallel["fitness"],
                "max_abs_field_difference": local_max,
                "match": match,
            }
        )

    if len(parallel_rows) != len(serial_rows):
        failures.append(
            f"parallel result count {len(parallel_rows)} != serial count {len(serial_rows)}"
        )

    compare_path = out / "parallel_reproduction.csv"
    with compare_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)

    seconds_per_mask = wall_seconds / len(masks)
    one_split_hours = seconds_per_mask * 65535 / 3600.0
    all_20_runner_hours = one_split_hours * 20
    effective_cpu_percent = (
        100.0 * child_cpu_seconds / wall_seconds if wall_seconds > 0 else None
    )

    warning_counts = Counter(w.category.__name__ for w in caught)

    summary = {
        "phase": 11,
        "work_package": "B",
        "parallel_validation_passed": not failures,
        "workers": args.workers,
        "seed": args.seed,
        "masks": len(masks),
        "wall_seconds": wall_seconds,
        "child_cpu_seconds": child_cpu_seconds,
        "seconds_per_mask_wall": seconds_per_mask,
        "effective_child_cpu_percent": effective_cpu_percent,
        "max_abs_difference_vs_serial": max_diff,
        "projected_parallel_wall_hours_per_split": one_split_hours,
        "projected_aggregate_runner_hours_20_splits": all_20_runner_hours,
        "warning_counts": dict(sorted(warning_counts.items())),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "thread_environment": {
                k: os.environ.get(k)
                for k in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
        },
        "failures": failures,
    }
    (out / "parallel_benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = [
        "# Phase 11 Work Package B — Process-level Parallel Benchmark",
        "",
        f"- Validation: **{'PASS' if not failures else 'FAIL'}**",
        f"- Workers: {args.workers}",
        f"- Masks rescored: {len(masks)}",
        f"- Maximum absolute difference versus serial scores: {max_diff:.3e}",
        f"- Parallel wall time: {wall_seconds:.2f} s",
        f"- Effective child CPU use: {effective_cpu_percent:.1f}%" if effective_cpu_percent is not None else "- Effective child CPU use: unavailable",
        f"- Projected wall time per complete split at this measured rate: {one_split_hours:.2f} h",
        f"- Aggregate runner-hours for 20 splits at this measured rate: {all_20_runner_hours:.2f} h",
        "",
        "## Interpretation",
        "",
        "The parallel path changes only execution scheduling. Each worker creates the same frozen "
        "SubsetEvaluator for the same outer-training data and seed. The benchmark re-scores the exact "
        "1,000 masks used by the serial feasibility benchmark and compares every fold score, CV mean, "
        "and penalized fitness.",
        "",
        "Full exhaustive enumeration remains blocked until Gate B is explicitly recorded in PHASE11_STATUS.md.",
        "",
        "## Failures",
        "",
    ]
    report.extend(f"- {x}" for x in failures) if failures else report.append("- None.")
    (out / "PARALLEL_BENCHMARK_REPORT.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
