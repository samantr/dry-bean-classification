#!/usr/bin/env python3
"""Phase 11 Work Package E: exhaustive exact-objective enumeration for one outer split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import tempfile
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

DIMENSION = 16
TOTAL_MASKS = (1 << DIMENSION) - 1
PENALTY = 0.001
TIE_ATOL = 1e-12
_WORKER_EVALUATOR = None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def atomic_json(path: Path, payload: dict) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_split(dataset: Path, seed: int):
    frame = pd.read_excel(dataset)
    if "Class" not in frame.columns:
        raise ValueError("Dataset is missing Class target")
    features = frame.drop(columns="Class")
    if features.shape[1] != DIMENSION:
        raise ValueError(f"Expected {DIMENSION} inputs, got {features.shape[1]}")
    X = features.to_numpy(dtype=float)
    y = LabelEncoder().fit_transform(frame["Class"])
    train, test = train_test_split(
        np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed
    )
    return frame, features, X[train], y[train], train, test


def init_worker(X_train, y_train, seed):
    global _WORKER_EVALUATOR
    _WORKER_EVALUATOR = SubsetEvaluator(X_train, y_train, seed, penalty=PENALTY)


def score_mask(mask: int) -> dict:
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r = _WORKER_EVALUATOR.evaluate(int(mask))
        return {
            "mask": int(mask),
            "selected_count": int(r["selected_count"]),
            "cv_macro_f1": float(r["cv_macro_f1"]),
            "fitness": float(r["fitness"]),
            "fold1": float(r["fold_scores"][0]),
            "fold2": float(r["fold_scores"][1]),
            "fold3": float(r["fold_scores"][2]),
            "status": "ok",
            "warnings": "|".join(sorted({w.category.__name__ for w in caught})),
            "error": "",
        }
    except Exception as exc:  # scientific archive must record rather than silently drop failures
        return {
            "mask": int(mask),
            "selected_count": "",
            "cv_macro_f1": "",
            "fitness": "",
            "fold1": "",
            "fold2": "",
            "fold3": "",
            "status": "error",
            "warnings": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


FIELDS = [
    "mask", "selected_count", "cv_macro_f1", "fitness",
    "fold1", "fold2", "fold3", "status", "warnings", "error"
]


def validate_partition(path: Path, start_mask: int, end_mask: int) -> dict:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected = list(range(start_mask, end_mask + 1))
    masks = [int(r["mask"]) for r in rows]
    failures = []
    if masks != expected:
        failures.append("mask range/order mismatch")
    if len(masks) != len(set(masks)):
        failures.append("duplicate masks")
    if any(m == 0 for m in masks):
        failures.append("zero mask")
    errors = [r for r in rows if r["status"] != "ok"]
    if errors:
        failures.append(f"{len(errors)} failed evaluations")
    for r in rows:
        if r["status"] == "ok":
            values = [float(r[k]) for k in ("cv_macro_f1", "fitness", "fold1", "fold2", "fold3")]
            if not all(math.isfinite(v) for v in values):
                failures.append(f"nonfinite score mask={r['mask']}")
                break
            count = int(r["selected_count"])
            if count != int(r["mask"]).bit_count():
                failures.append(f"cardinality mismatch mask={r['mask']}")
                break
            expected_fit = float(r["cv_macro_f1"]) - PENALTY * count / DIMENSION
            if not math.isclose(float(r["fitness"]), expected_fit, rel_tol=0.0, abs_tol=TIE_ATOL):
                failures.append(f"fitness mismatch mask={r['mask']}")
                break
    return {"rows": len(rows), "failures": failures, "sha256": sha256_file(path)}


def write_partition_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def choose_optimum(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["status"] == "ok"]
    best_fit = max(float(r["fitness"]) for r in ok)
    c1 = [r for r in ok if abs(float(r["fitness"]) - best_fit) <= TIE_ATOL]
    best_cv = max(float(r["cv_macro_f1"]) for r in c1)
    c2 = [r for r in c1 if abs(float(r["cv_macro_f1"]) - best_cv) <= TIE_ATOL]
    min_count = min(int(r["selected_count"]) for r in c2)
    c3 = [r for r in c2 if int(r["selected_count"]) == min_count]
    return min(c3, key=lambda r: int(r["mask"]))


def best_by_cardinality(rows: list[dict]) -> list[dict]:
    result = []
    for count in range(1, DIMENSION + 1):
        c = [r for r in rows if r["status"] == "ok" and int(r["selected_count"]) == count]
        best_cv = max(float(r["cv_macro_f1"]) for r in c)
        tied = [r for r in c if abs(float(r["cv_macro_f1"]) - best_cv) <= TIE_ATOL]
        best = min(tied, key=lambda r: int(r["mask"]))
        result.append({
            "selected_count": count,
            "mask": int(best["mask"]),
            "cv_macro_f1": float(best["cv_macro_f1"]),
            "fitness": float(best["fitness"]),
        })
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--partition-size", type=int, default=4096)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if not 1001 <= args.seed <= 1020:
        p.error("seed must be one of the frozen outer seeds 1001..1020")
    if args.workers < 1:
        p.error("workers must be positive")
    if not 1 <= args.partition_size <= TOTAL_MASKS:
        p.error("invalid partition size")

    out = args.output_dir
    if out.exists() and not args.resume:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    parts = out / "partitions"
    parts.mkdir(exist_ok=True)

    frame, features, X_train, y_train, train_idx, test_idx = load_split(args.dataset, args.seed)
    started = time.perf_counter()
    partition_meta = []
    warning_counts = Counter()
    all_rows = []

    for start in range(1, TOTAL_MASKS + 1, args.partition_size):
        end = min(TOTAL_MASKS, start + args.partition_size - 1)
        path = parts / f"masks-{start:05d}-{end:05d}.csv"
        if args.resume and path.exists():
            check = validate_partition(path, start, end)
            if not check["failures"]:
                with path.open(newline="", encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream))
                all_rows.extend(rows)
                partition_meta.append({"start": start, "end": end, **check, "resumed": True})
                continue
            path.unlink()

        masks = list(range(start, end + 1))
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=init_worker,
            initargs=(X_train, y_train, args.seed),
        ) as pool:
            rows = list(pool.map(score_mask, masks, chunksize=8))
        write_partition_atomic(path, rows)
        check = validate_partition(path, start, end)
        if check["failures"]:
            atomic_json(out / "failure.json", {"seed": args.seed, "partition": [start, end], "failures": check["failures"]})
            raise RuntimeError(f"partition validation failed {start}-{end}: {check['failures']}")
        for r in rows:
            if r["warnings"]:
                for name in r["warnings"].split("|"):
                    if name:
                        warning_counts[name] += 1
        all_rows.extend(rows)
        partition_meta.append({"start": start, "end": end, **check, "resumed": False})

    masks = [int(r["mask"]) for r in all_rows]
    failures = []
    if len(all_rows) != TOTAL_MASKS:
        failures.append(f"row count {len(all_rows)} != {TOTAL_MASKS}")
    if set(masks) != set(range(1, TOTAL_MASKS + 1)):
        failures.append("mask coverage is not exactly 1..65535")
    if len(set(masks)) != TOTAL_MASKS:
        failures.append("duplicate masks in merged archive")
    bad = [r for r in all_rows if r["status"] != "ok"]
    if bad:
        failures.append(f"{len(bad)} failed mask evaluations")
    if failures:
        atomic_json(out / "failure.json", {"seed": args.seed, "failures": failures})
        raise RuntimeError("; ".join(failures))

    optimum = choose_optimum(all_rows)
    card = best_by_cardinality(all_rows)
    elapsed = time.perf_counter() - started

    with (out / "exact_optimum.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["seed", "mask", "mask_bits", "selected_count", "cv_macro_f1", "fitness"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "seed": args.seed,
            "mask": int(optimum["mask"]),
            "mask_bits": format(int(optimum["mask"]), "016b"),
            "selected_count": int(optimum["selected_count"]),
            "cv_macro_f1": float(optimum["cv_macro_f1"]),
            "fitness": float(optimum["fitness"]),
        })

    with (out / "best_by_cardinality.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["seed", "selected_count", "mask", "mask_bits", "cv_macro_f1", "fitness"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in card:
            writer.writerow({"seed": args.seed, "mask_bits": format(r["mask"], "016b"), **r})

    mapping = [{"bit": i, "feature_name": str(features.columns[i])} for i in range(DIMENSION)]
    with (out / "feature_mapping.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["bit", "feature_name"])
        writer.writeheader(); writer.writerows(mapping)

    manifest = {
        "phase": 11,
        "work_package": "E",
        "seed": args.seed,
        "state": "complete",
        "nonempty_masks": TOTAL_MASKS,
        "inner_model_fits": TOTAL_MASKS * 3,
        "workers": args.workers,
        "partition_size": args.partition_size,
        "partition_count": len(partition_meta),
        "elapsed_wall_seconds": elapsed,
        "warnings_by_mask_category": dict(sorted(warning_counts.items())),
        "dataset_sha256": sha256_file(args.dataset),
        "budgeted_search_sha256": sha256_file(ROOT / "src" / "budgeted_search.py"),
        "outer_train_indices_sha256": sha256_bytes(np.asarray(train_idx, dtype=np.int64).tobytes()),
        "outer_test_indices_sha256": sha256_bytes(np.asarray(test_idx, dtype=np.int64).tobytes()),
        "outer_test_used_for_search": False,
        "objective": "mean 3-fold inner macro-F1 - 0.001 * cardinality / 16",
        "tie_atol": TIE_ATOL,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "thread_environment": {k: os.environ.get(k) for k in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"
            )},
        },
        "partitions": partition_meta,
    }
    atomic_json(out / "manifest.json", manifest)

    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "sha256sums.txt")
    atomic_text(out / "sha256sums.txt", "".join(f"{sha256_file(p)}  {p.relative_to(out)}\n" for p in files))
    print(json.dumps({
        "seed": args.seed,
        "state": "complete",
        "elapsed_wall_seconds": elapsed,
        "optimum_mask": int(optimum["mask"]),
        "optimum_fitness": float(optimum["fitness"]),
        "optimum_cv_macro_f1": float(optimum["cv_macro_f1"]),
        "selected_count": int(optimum["selected_count"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
