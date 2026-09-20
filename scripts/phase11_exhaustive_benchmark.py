#!/usr/bin/env python3
"""Phase 11 Work Package B: exact-objective validation and exhaustive-search feasibility benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import resource
import statistics
import sys
import tarfile
import tempfile
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from budgeted_search import SubsetEvaluator, mask_indices  # noqa: E402


DIMENSION = 16
TOTAL_MASKS = (1 << DIMENSION) - 1
PENALTY = 0.001


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if destination != target and destination not in target.parents:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
        tar.extractall(destination)


def find_archived_result(archive: Path, seed: int, method: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="phase11-benchmark-archive-") as tmp:
        root = Path(tmp)
        safe_extract(archive, root)
        for path in root.rglob("result.json"):
            result = json.loads(path.read_text(encoding="utf-8"))
            if int(result.get("seed", -1)) == seed and result.get("method") == method:
                return result
    raise FileNotFoundError(f"Archived result not found for seed={seed}, method={method}")


def make_outer_training(dataset: Path, seed: int):
    frame = pd.read_excel(dataset)
    if "Class" not in frame.columns:
        raise ValueError("Dataset is missing Class target")
    features = frame.drop(columns="Class")
    X = features.to_numpy(dtype=float)
    y = LabelEncoder().fit_transform(frame["Class"])
    train, test = train_test_split(
        np.arange(len(y)),
        test_size=0.2,
        stratify=y,
        random_state=seed,
    )
    return frame, features, X[train], y[train], train, test


def record_for_csv(record: dict) -> dict:
    folds = list(record["fold_scores"])
    return {
        "mask": int(record["mask"]),
        "selected_count": int(record["selected_count"]),
        "cv_macro_f1": float(record["cv_macro_f1"]),
        "fitness": float(record["fitness"]),
        "fold1": float(folds[0]),
        "fold2": float(folds[1]),
        "fold3": float(folds[2]),
    }


def write_records(path: Path, records: list[dict]) -> int:
    rows = [record_for_csv(r) for r in records]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path.stat().st_size


def score_masks(evaluator: SubsetEvaluator, masks: list[int]):
    captured = []
    started_wall = time.perf_counter()
    before = resource.getrusage(resource.RUSAGE_SELF)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for mask in masks:
            evaluator.evaluate(mask)
        captured.extend(caught)
    after = resource.getrusage(resource.RUSAGE_SELF)
    elapsed_wall = time.perf_counter() - started_wall
    cpu_seconds = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
    warning_counts = Counter(
        "ConvergenceWarning" if issubclass(w.category, ConvergenceWarning) else w.category.__name__
        for w in captured
    )
    return evaluator.history, elapsed_wall, cpu_seconds, dict(sorted(warning_counts.items()))


def choose_archived_validation_rows(history: list[dict], max_rows: int = 24) -> list[dict]:
    selected = []
    seen_masks = set()
    seen_cardinalities = set()

    for row in history[:20]:
        mask = int(row["mask"])
        if mask not in seen_masks:
            selected.append(row)
            seen_masks.add(mask)
            seen_cardinalities.add(int(row["selected_count"]))

    for row in history:
        count = int(row["selected_count"])
        mask = int(row["mask"])
        if count not in seen_cardinalities and mask not in seen_masks:
            selected.append(row)
            seen_masks.add(mask)
            seen_cardinalities.add(count)
        if len(selected) >= max_rows:
            break

    best = max(
        history,
        key=lambda r: (float(r["fitness"]), -int(r["selected_count"]), -int(r["mask"])),
    )
    if int(best["mask"]) not in seen_masks:
        selected.append(best)
    return selected


def compare_records(expected: dict, actual: dict, atol: float) -> list[str]:
    failures = []
    if int(expected["mask"]) != int(actual["mask"]):
        failures.append("mask")
    if int(expected["selected_count"]) != int(actual["selected_count"]):
        failures.append("selected_count")
    for key in ("cv_macro_f1", "fitness"):
        if not math.isclose(
            float(expected[key]), float(actual[key]), rel_tol=0.0, abs_tol=atol
        ):
            failures.append(key)
    expected_folds = np.asarray(expected["fold_scores"], dtype=float)
    actual_folds = np.asarray(actual["fold_scores"], dtype=float)
    if expected_folds.shape != (3,) or actual_folds.shape != (3,):
        failures.append("fold_count")
    elif not np.allclose(expected_folds, actual_folds, rtol=0.0, atol=atol):
        failures.append("fold_scores")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--archive-method", default="random_search")
    parser.add_argument("--benchmark-masks", type=int, default=1000)
    parser.add_argument("--warmup-masks", type=int, default=32)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--partition-size", type=int, default=4096)
    args = parser.parse_args()

    if not 500 <= args.benchmark_masks <= 1000:
        parser.error("--benchmark-masks must be between 500 and 1000")
    if not 1 <= args.warmup_masks < args.benchmark_masks:
        parser.error("--warmup-masks must be positive and smaller than benchmark size")
    if not 1 <= args.partition_size <= TOTAL_MASKS:
        parser.error("invalid partition size")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    frame, features, X_train, y_train, outer_train, outer_test = make_outer_training(
        args.dataset, args.seed
    )
    archived = find_archived_result(args.archive, args.seed, args.archive_method)
    history = archived["search"]["history"]
    archived_rows = choose_archived_validation_rows(history)

    if archived["train_indices"] != outer_train.tolist():
        failures.append("archived outer-training indices do not reproduce for benchmark seed")
    if archived["test_indices"] != outer_test.tolist():
        failures.append("archived outer-test indices do not reproduce for benchmark seed")

    bit_mapping = []
    for bit in range(DIMENSION):
        mask = 1 << bit
        indices = mask_indices(mask, DIMENSION)
        ok = indices == [bit]
        if not ok:
            failures.append(f"bit mapping failed at bit {bit}")
        bit_mapping.append(
            {
                "bit": bit,
                "mask": mask,
                "feature_index": indices[0] if indices else "",
                "feature_name": features.columns[bit],
                "roundtrip_ok": ok,
            }
        )
    with (out / "bit_feature_mapping.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(bit_mapping[0]))
        writer.writeheader()
        writer.writerows(bit_mapping)

    validation_eval = SubsetEvaluator(X_train, y_train, args.seed, penalty=PENALTY)
    validation_actual = []
    validation_rows = []
    with warnings.catch_warnings(record=True) as validation_warnings:
        warnings.simplefilter("always")
        for expected in archived_rows:
            actual = validation_eval.evaluate(int(expected["mask"]))
            validation_actual.append(actual)
            differences = compare_records(expected, actual, args.atol)
            if differences:
                failures.append(
                    f"archived score mismatch for mask {expected['mask']}: {','.join(differences)}"
                )
            validation_rows.append(
                {
                    "mask": int(expected["mask"]),
                    "selected_count": int(expected["selected_count"]),
                    "archived_cv_macro_f1": float(expected["cv_macro_f1"]),
                    "recomputed_cv_macro_f1": float(actual["cv_macro_f1"]),
                    "abs_cv_diff": abs(
                        float(expected["cv_macro_f1"]) - float(actual["cv_macro_f1"])
                    ),
                    "archived_fitness": float(expected["fitness"]),
                    "recomputed_fitness": float(actual["fitness"]),
                    "abs_fitness_diff": abs(float(expected["fitness"]) - float(actual["fitness"])),
                    "match": not differences,
                }
            )

    with (out / "archived_score_reproduction.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(validation_rows[0]))
        writer.writeheader()
        writer.writerows(validation_rows)

    single_full_masks = [1 << bit for bit in range(DIMENSION)] + [TOTAL_MASKS]
    single_eval = SubsetEvaluator(X_train, y_train, args.seed, penalty=PENALTY)
    single_records, _, _, single_warnings = score_masks(single_eval, single_full_masks)
    write_records(out / "single_and_full_mask_scores.csv", single_records)

    for record in single_records:
        mask = int(record["mask"])
        indices = mask_indices(mask, DIMENSION)
        roundtrip = sum(1 << i for i in indices)
        if roundtrip != mask:
            failures.append(f"mask-feature roundtrip failed for mask {mask}")
        if not math.isfinite(float(record["fitness"])):
            failures.append(f"non-finite validation fitness for mask {mask}")

    all_masks = list(range(1, TOTAL_MASKS + 1))
    random.Random(20260920 + args.seed).shuffle(all_masks)
    excluded = {int(r["mask"]) for r in validation_actual}
    warmup = []
    benchmark = []
    for mask in all_masks:
        if mask in excluded:
            continue
        if len(warmup) < args.warmup_masks:
            warmup.append(mask)
            continue
        if len(benchmark) < args.benchmark_masks:
            benchmark.append(mask)
        if len(benchmark) == args.benchmark_masks:
            break

    warmup_eval = SubsetEvaluator(X_train, y_train, args.seed, penalty=PENALTY)
    score_masks(warmup_eval, warmup)

    benchmark_eval = SubsetEvaluator(X_train, y_train, args.seed, penalty=PENALTY)
    bench_records, wall_seconds, cpu_seconds, bench_warnings = score_masks(
        benchmark_eval, benchmark
    )
    benchmark_csv = out / "benchmark_sample.csv"
    benchmark_bytes = write_records(benchmark_csv, bench_records)

    if len(benchmark_eval.cache) != args.benchmark_masks:
        failures.append("benchmark did not contain the requested number of uncached unique masks")
    if len({int(r["mask"]) for r in bench_records}) != args.benchmark_masks:
        failures.append("duplicate benchmark mask detected")

    seconds_per_mask = wall_seconds / args.benchmark_masks
    cpu_seconds_per_mask = cpu_seconds / args.benchmark_masks
    projected_one_split_seconds = seconds_per_mask * TOTAL_MASKS
    projected_all_splits_seconds = projected_one_split_seconds * 20

    with benchmark_csv.open("rb") as stream:
        line_sizes = [len(line) for line in stream.readlines()[1:]]
    mean_csv_bytes_per_mask = statistics.fmean(line_sizes)
    projected_csv_bytes_per_split = mean_csv_bytes_per_mask * TOTAL_MASKS
    projected_csv_bytes_all_splits = projected_csv_bytes_per_split * 20

    partitions = []
    start = 1
    partition_id = 0
    while start <= TOTAL_MASKS:
        end = min(TOTAL_MASKS, start + args.partition_size - 1)
        partitions.append(
            {
                "partition": partition_id,
                "mask_start_inclusive": start,
                "mask_end_inclusive": end,
                "mask_count": end - start + 1,
                "atomic_output_name": f"part-{partition_id:03d}.csv",
            }
        )
        start = end + 1
        partition_id += 1
    (out / "mask_partition_plan.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "total_masks": TOTAL_MASKS,
                "partition_size": args.partition_size,
                "partitions": partitions,
                "resume_rule": (
                    "Each partition is scored independently with the frozen SubsetEvaluator, "
                    "written to a temporary file, validated for exact mask coverage/uniqueness/finite "
                    "scores, then atomically renamed. Resume skips only validated final partition files. "
                    "A split is exact only after all partitions pass and the merged mask set is exactly 1..65535."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    validation_warning_counts = Counter(
        "ConvergenceWarning" if issubclass(w.category, ConvergenceWarning) else w.category.__name__
        for w in validation_warnings
    )
    warning_counts = Counter(single_warnings)
    warning_counts.update(bench_warnings)
    warning_counts.update(validation_warning_counts)

    summary = {
        "phase": 11,
        "work_package": "B",
        "gate_b_ready_for_decision": not failures,
        "full_enumeration_launched": False,
        "benchmark_seed": args.seed,
        "archive_method_for_reproduction": args.archive_method,
        "dataset": {
            "sha256": sha256_file(args.dataset),
            "rows": int(len(frame)),
            "input_features": int(len(features.columns)),
            "feature_names": features.columns.tolist(),
        },
        "objective": {
            "dimension": DIMENSION,
            "nonempty_masks_per_split": TOTAL_MASKS,
            "inner_folds_per_mask": 3,
            "inner_model_fits_per_split": 3 * TOTAL_MASKS,
            "inner_model_fits_all_20_splits": 3 * TOTAL_MASKS * 20,
            "penalty": PENALTY,
            "scorer": "frozen SubsetEvaluator from src/budgeted_search.py",
        },
        "validation": {
            "archived_masks_recomputed": len(validation_rows),
            "all_archived_scores_match": all(r["match"] for r in validation_rows),
            "max_abs_cv_macro_f1_difference": max(r["abs_cv_diff"] for r in validation_rows),
            "max_abs_fitness_difference": max(r["abs_fitness_diff"] for r in validation_rows),
            "single_feature_masks_scored": DIMENSION,
            "full_feature_mask_scored": True,
            "bit_mapping_roundtrip_passed": all(r["roundtrip_ok"] for r in bit_mapping),
        },
        "benchmark": {
            "warmup_masks": args.warmup_masks,
            "uncached_masks": args.benchmark_masks,
            "wall_seconds": wall_seconds,
            "cpu_seconds": cpu_seconds,
            "seconds_per_mask": seconds_per_mask,
            "cpu_seconds_per_mask": cpu_seconds_per_mask,
            "effective_single_process_cpu_percent": (
                100.0 * cpu_seconds / wall_seconds if wall_seconds > 0 else None
            ),
            "cardinality_distribution": dict(
                sorted(Counter(int(r["selected_count"]) for r in bench_records).items())
            ),
            "warning_counts": dict(sorted(warning_counts.items())),
        },
        "projection_serial": {
            "one_split_seconds": projected_one_split_seconds,
            "one_split_hours": projected_one_split_seconds / 3600.0,
            "all_20_splits_seconds": projected_all_splits_seconds,
            "all_20_splits_hours": projected_all_splits_seconds / 3600.0,
        },
        "disk_estimate_csv": {
            "benchmark_csv_bytes": benchmark_bytes,
            "mean_bytes_per_mask_row": mean_csv_bytes_per_mask,
            "one_split_bytes": projected_csv_bytes_per_split,
            "one_split_megabytes": projected_csv_bytes_per_split / (1024**2),
            "all_20_splits_bytes": projected_csv_bytes_all_splits,
            "all_20_splits_megabytes": projected_csv_bytes_all_splits / (1024**2),
            "note": (
                "Projection covers compact per-mask scientific CSV fields only. "
                "Manifests, logs, partitions, and hashes add modest overhead."
            ),
        },
        "resume_strategy": {
            "partition_size": args.partition_size,
            "partition_count_per_split": len(partitions),
            "atomic_partitions": True,
            "validation_before_commit": [
                "expected contiguous mask range",
                "no duplicate masks",
                "no zero mask",
                "finite fold scores/cv_macro_f1/fitness",
                "selected_count equals mask cardinality",
                "fitness formula reproduces",
            ],
        },
        "runtime_environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "logical_cpu_count": os.cpu_count(),
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
        },
        "failures": failures,
    }
    (out / "exhaustive_benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = [
        "# Phase 11 Work Package B — Exhaustive-search Feasibility Benchmark",
        "",
        f"- Validation status: **{'PASS' if not failures else 'FAIL'}**",
        "- Full exhaustive enumeration launched: **NO**",
        f"- Benchmark split seed: {args.seed}",
        f"- Archived Stage-3 masks recomputed: {len(validation_rows)}",
        f"- Maximum archived CV macro-F1 absolute difference: {summary['validation']['max_abs_cv_macro_f1_difference']:.3e}",
        f"- Maximum archived fitness absolute difference: {summary['validation']['max_abs_fitness_difference']:.3e}",
        f"- Benchmark uncached masks: {args.benchmark_masks}",
        f"- Benchmark wall time: {wall_seconds:.2f} s",
        f"- Mean wall time per mask: {seconds_per_mask:.6f} s",
        f"- Projected serial time for one 65,535-mask split: {projected_one_split_seconds / 3600.0:.2f} h",
        f"- Projected serial time for 20 splits: {projected_all_splits_seconds / 3600.0:.2f} h",
        f"- Estimated compact CSV size per split: {projected_csv_bytes_per_split / (1024**2):.1f} MiB",
        f"- Estimated compact CSV size for 20 splits: {projected_csv_bytes_all_splits / (1024**2):.1f} MiB",
        f"- Partition plan: {len(partitions)} atomic parts/split of at most {args.partition_size} masks",
        "",
        "## Exact-score validation",
        "",
        "The benchmark uses the same SubsetEvaluator implementation as frozen Stage 3. "
        "All sampled archived masks are recomputed on the same outer-training split and compared "
        "against archived fold scores, CV macro-F1, cardinality, and penalized fitness.",
        "",
        "All 16 single-feature masks and the full 16-feature mask are also scored to validate "
        "bit ordering, mask-to-feature mapping, nonempty-mask handling, and finite objective values.",
        "",
        "## Resume strategy",
        "",
        "Full enumeration, if approved after Gate B, will be partitioned by deterministic contiguous "
        "mask ranges. Each partition is written to a temporary file, validated, and atomically renamed. "
        "Resume will skip only validated final partitions. A split will not be called exact unless the "
        "merged mask set is exactly 1..65,535 with no duplicates, failures, or nonfinite scores.",
        "",
        "## Failures",
        "",
    ]
    report.extend(f"- {item}" for item in failures) if failures else report.append("- None.")
    report += [
        "",
        "## Gate B note",
        "",
        "This report records the required serial feasibility evidence. It does not authorize or launch "
        "the full 20-split exhaustive computation. The next decision must consider the measured runtime, "
        "disk estimate, exact-score reproduction, and whether safe process-level parallelism should be "
        "benchmarked before committing substantial GitHub Actions compute.",
        "",
    ]
    (out / "EXHAUSTIVE_BENCHMARK_REPORT.md").write_text(
        "\n".join(report), encoding="utf-8"
    )

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
