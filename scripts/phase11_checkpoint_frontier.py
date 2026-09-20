#!/usr/bin/env python3
"""Phase 11 Work Packages C/D: budget checkpoints and training-only performance-cardinality frontiers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tarfile
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from baseline_models import CLASSIFIERS  # noqa: E402
from budgeted_search import SEARCH_METHODS, mask_indices  # noqa: E402

CHECKPOINTS = (20, 50, 100, 150, 200)
TIE_ATOL = 1e-12
F1_TOLERANCES = (0.001, 0.005)
METHODS = tuple(SEARCH_METHODS)


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


def load_units(root: Path) -> dict[tuple[int, str], Path]:
    units = {}
    for path in root.rglob("result.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        method = data.get("method")
        seed = data.get("seed")
        if method in METHODS and seed is not None:
            key = (int(seed), str(method))
            if key in units:
                raise RuntimeError(f"duplicate archived unit {key}")
            units[key] = path
    expected = {(s, m) for s in range(1001, 1021) for m in METHODS}
    missing = sorted(expected - set(units))
    if missing:
        raise RuntimeError(f"missing {len(missing)} search units, first={missing[:3]}")
    return units


def checkpoint_best(history: list[dict], budget: int) -> dict:
    prefix = history[:budget]
    if len(prefix) != budget:
        raise ValueError(f"history shorter than checkpoint {budget}")
    evaluations = [int(r.get("evaluation", i + 1)) for i, r in enumerate(prefix)]
    if evaluations != list(range(1, budget + 1)):
        raise ValueError(f"checkpoint prefix is not evaluations 1..{budget}")
    best_fit = max(float(r["fitness"]) for r in prefix)
    c1 = [r for r in prefix if abs(float(r["fitness"]) - best_fit) <= TIE_ATOL]
    best_cv = max(float(r["cv_macro_f1"]) for r in c1)
    c2 = [r for r in c1 if abs(float(r["cv_macro_f1"]) - best_cv) <= TIE_ATOL]
    min_count = min(int(r["selected_count"]) for r in c2)
    c3 = [r for r in c2 if int(r["selected_count"]) == min_count]
    return min(c3, key=lambda r: int(r["mask"]))


def dominates(a: dict, b: dict) -> bool:
    ac, bc = int(a["selected_count"]), int(b["selected_count"])
    av, bv = float(a["cv_macro_f1"]), float(b["cv_macro_f1"])
    weak = ac <= bc and av >= bv - TIE_ATOL
    strict = ac < bc or av > bv + TIE_ATOL
    return weak and strict


def nondominated(history: list[dict]) -> list[dict]:
    result = []
    for i, row in enumerate(history):
        if not any(i != j and dominates(other, row) for j, other in enumerate(history)):
            result.append(row)
    return result


def coverage(a: list[dict], b: list[dict]) -> float:
    return sum(any(dominates(x, y) for x in a) for y in b) / len(b)


def split_data(dataset: Path, seed: int):
    frame = pd.read_excel(dataset)
    features = frame.drop(columns="Class")
    X = features.to_numpy(dtype=float)
    y = LabelEncoder().fit_transform(frame["Class"])
    train, test = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
    return features, X, y, train, test


def evaluate_mask(X, y, train, test, mask: int, seed: int) -> tuple[list[dict], dict[str, np.ndarray]]:
    selected = mask_indices(mask, X.shape[1])
    X_train = X[train][:, selected]
    X_test = X[test][:, selected]
    y_train, y_test = y[train], y[test]
    rows = []
    predictions = {}
    for name, factory in CLASSIFIERS.items():
        model = factory(seed)
        if name != "Random Forest":
            model = make_pipeline(StandardScaler(), model)
        start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_s = time.perf_counter() - start
        start = time.perf_counter()
        pred = model.predict(X_test)
        pred_s = time.perf_counter() - start
        rows.append({
            "classifier": name,
            "macro_f1": float(f1_score(y_test, pred, average="macro")),
            "accuracy": float(accuracy_score(y_test, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
            "fit_seconds": fit_s,
            "predict_seconds": pred_s,
        })
        predictions[name] = np.asarray(pred, dtype=np.int16)
    return rows, predictions


def analyze_seed(seed: int, dataset: str, unit_paths: dict[str, str]):
    features, X, y, train, test = split_data(Path(dataset), seed)
    checkpoint_rows = []
    frontier_rows = []
    tolerance_rows = []
    coverage_rows = []
    prediction_arrays = {}
    histories = {}
    validation_failures = []

    for method in METHODS:
        archived = json.loads(Path(unit_paths[method]).read_text(encoding="utf-8"))
        if archived["train_indices"] != train.tolist() or archived["test_indices"] != test.tolist():
            validation_failures.append(f"{seed}-{method}: outer split mismatch")
        history = archived["search"]["history"]
        if len(history) != 200 or len({int(r["mask"]) for r in history}) != 200:
            validation_failures.append(f"{seed}-{method}: history is not 200 unique masks")
        histories[method] = history

        front = nondominated(history)
        front_masks = {int(r["mask"]) for r in front}
        best_trace_cv = max(float(r["cv_macro_f1"]) for r in history)
        observed_counts = sorted({int(r["selected_count"]) for r in history})
        for count in observed_counts:
            candidates = [r for r in history if int(r["selected_count"]) == count]
            best = max(candidates, key=lambda r: (float(r["cv_macro_f1"]), -int(r["mask"])))
            frontier_rows.append({
                "seed": seed, "method": method, "selected_count": count,
                "best_cv_macro_f1_at_cardinality": float(best["cv_macro_f1"]),
                "best_mask_at_cardinality": int(best["mask"]),
                "nondominated_masks_at_cardinality": sum(
                    1 for r in front if int(r["selected_count"]) == count
                ),
            })
        for tol in F1_TOLERANCES:
            eligible = [r for r in history if float(r["cv_macro_f1"]) >= best_trace_cv - tol - TIE_ATOL]
            tolerance_rows.append({
                "seed": seed, "method": method, "absolute_f1_tolerance": tol,
                "trace_best_cv_macro_f1": best_trace_cv,
                "smallest_cardinality_within_tolerance": min(int(r["selected_count"]) for r in eligible),
            })

        for budget in CHECKPOINTS:
            chosen = checkpoint_best(history, budget)
            eval_rows, preds = evaluate_mask(X, y, train, test, int(chosen["mask"]), seed)
            for metrics in eval_rows:
                row = {
                    "seed": seed, "method": method, "budget": budget,
                    "mask": int(chosen["mask"]),
                    "mask_bits": format(int(chosen["mask"]), "016b"),
                    "selected_count": int(chosen["selected_count"]),
                    "inner_cv_macro_f1": float(chosen["cv_macro_f1"]),
                    "fitness": float(chosen["fitness"]),
                    "feature_names": "|".join(features.columns[mask_indices(int(chosen["mask"]), 16)].tolist()),
                    "cumulative_search_seconds": "",
                    **metrics,
                }
                checkpoint_rows.append(row)
                key = f"s{seed}_{method}_b{budget}_{metrics['classifier'].lower().replace(' ', '_')}"
                prediction_arrays[key] = preds[metrics["classifier"]]

            if budget == 200:
                if int(chosen["mask"]) != int(archived["search"]["best"]["mask"]):
                    validation_failures.append(f"{seed}-{method}: budget-200 selected mask mismatch")
                for metrics in eval_rows:
                    expected = archived["classifiers"][metrics["classifier"]]
                    for field in ("macro_f1", "accuracy", "balanced_accuracy"):
                        if not math.isclose(float(metrics[field]), float(expected[field]), rel_tol=0.0, abs_tol=TIE_ATOL):
                            validation_failures.append(
                                f"{seed}-{method}-{metrics['classifier']}: budget-200 {field} mismatch"
                            )
                    if not np.array_equal(preds[metrics["classifier"]],
                                          np.asarray(expected["predictions"], dtype=np.int16)):
                        validation_failures.append(
                            f"{seed}-{method}-{metrics['classifier']}: budget-200 predictions mismatch"
                        )

        frontier_rows.append({
            "seed": seed, "method": method, "selected_count": "",
            "best_cv_macro_f1_at_cardinality": "",
            "best_mask_at_cardinality": "",
            "nondominated_masks_at_cardinality": len(front_masks),
        })

    initial = {
        tuple(int(r["mask"]) for r in histories[m][:20]) for m in METHODS
    }
    if len(initial) != 1:
        validation_failures.append(f"{seed}: initial 20 masks differ across methods")
    # The identical first-20 candidates imply identical checkpoint selections at budget 20.
    first20 = {
        (int(checkpoint_best(histories[m], 20)["mask"]),
         float(checkpoint_best(histories[m], 20)["fitness"]))
        for m in METHODS
    }
    if len(first20) != 1:
        validation_failures.append(f"{seed}: budget-20 checkpoint differs despite shared initial masks")

    random_history = histories["random_search"]
    for method in METHODS:
        if method == "random_search":
            continue
        coverage_rows.append({
            "seed": seed,
            "ga_method": method,
            "coverage_ga_over_random": coverage(histories[method], random_history),
            "coverage_random_over_ga": coverage(random_history, histories[method]),
        })

    return {
        "seed": seed,
        "checkpoint_rows": checkpoint_rows,
        "frontier_rows": frontier_rows,
        "tolerance_rows": tolerance_rows,
        "coverage_rows": coverage_rows,
        "predictions": prediction_arrays,
        "validation_failures": validation_failures,
        "test_indices": test,
        "y_true": y[test],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="phase11-cd-") as tmp:
        root = Path(tmp)
        safe_extract(args.archive, root)
        units = load_units(root)
        task_args = []
        for seed in range(1001, 1021):
            task_args.append((seed, str(args.dataset), {m: str(units[(seed, m)]) for m in METHODS}))
        if args.workers == 1:
            results = [analyze_seed(*x) for x in task_args]
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(analyze_seed, *x) for x in task_args]
                results = [f.result() for f in futures]

    results.sort(key=lambda x: x["seed"])
    failures = [x for r in results for x in r["validation_failures"]]
    checkpoint_rows = [x for r in results for x in r["checkpoint_rows"]]
    frontier_rows = [x for r in results for x in r["frontier_rows"]]
    tolerance_rows = [x for r in results for x in r["tolerance_rows"]]
    coverage_rows = [x for r in results for x in r["coverage_rows"]]

    # Exclude per-trace sentinel rows from cardinality frequency.
    frontier_detail = [r for r in frontier_rows if r["selected_count"] != ""]
    frequency = []
    for method in METHODS:
        for count in range(1, 17):
            seeds_with = {
                int(r["seed"]) for r in frontier_detail
                if r["method"] == method and int(r["selected_count"]) == count
                and int(r["nondominated_masks_at_cardinality"]) > 0
            }
            frequency.append({
                "method": method, "selected_count": count,
                "splits_with_frontier_candidate": len(seeds_with),
                "fraction_of_20_splits": len(seeds_with) / 20.0,
            })

    # Descriptive best-so-far summaries only; no independent-sample inference.
    curve_summary = []
    for method in METHODS:
        for budget in CHECKPOINTS:
            # one row per classifier exists; use LR row because inner values are classifier-independent
            rows = [r for r in checkpoint_rows if r["method"] == method and int(r["budget"]) == budget
                    and r["classifier"] == "Logistic Regression"]
            vals = [float(r["fitness"]) for r in rows]
            cvs = [float(r["inner_cv_macro_f1"]) for r in rows]
            counts = [int(r["selected_count"]) for r in rows]
            curve_summary.append({
                "method": method, "budget": budget,
                "mean_best_fitness": float(np.mean(vals)),
                "median_best_fitness": float(np.median(vals)),
                "min_best_fitness": float(np.min(vals)),
                "max_best_fitness": float(np.max(vals)),
                "mean_inner_cv_macro_f1": float(np.mean(cvs)),
                "mean_selected_count": float(np.mean(counts)),
            })

    write_csv(out / "checkpoint_outer_results.csv", checkpoint_rows)
    write_csv(out / "checkpoint_curve_summary.csv", curve_summary)
    write_csv(out / "frontier_by_cardinality.csv", frontier_detail)
    write_csv(out / "frontier_tolerance_compactness.csv", tolerance_rows)
    write_csv(out / "dominance_coverage.csv", coverage_rows)
    write_csv(out / "frontier_cardinality_frequency.csv", frequency)

    arrays = {}
    for r in results:
        arrays[f"s{r['seed']}_test_indices"] = np.asarray(r["test_indices"], dtype=np.int32)
        arrays[f"s{r['seed']}_y_true"] = np.asarray(r["y_true"], dtype=np.int16)
        arrays.update(r["predictions"])
    np.savez_compressed(out / "checkpoint_predictions.npz", **arrays)

    report = [
        "# Phase 11 Work Packages C/D — Checkpoints and Frontiers",
        "",
        f"- Validation: **{'PASS' if not failures else 'FAIL'}**",
        "- Seeds: 1001–1020",
        "- Search methods: random search plus four frozen GA configurations",
        "- Checkpoints: 20, 50, 100, 150, 200 unique evaluations",
        "- Budget-20 role: protocol sanity check only; all methods share the first 20 masks within a seed.",
        "- Selection quantities: training-only frozen fitness and inner macro-F1.",
        "- Outer results: post-selection assessment only; never used to select checkpoint masks.",
        "- Frontier objectives: maximize inner macro-F1 and minimize original-feature cardinality.",
        f"- Fixed descriptive inner-F1 tolerances: {', '.join(map(str, F1_TOLERANCES))}.",
        "- Hypervolume: not used.",
        "- Per-evaluation timestamps: unavailable/trustworthy cumulative search time therefore not reconstructed.",
        "- Budget-200 outer predictions and metrics are required to reproduce archived Stage-3 values exactly.",
        "",
        "## Validation failures",
        "",
    ]
    report.extend(f"- {x}" for x in failures) if failures else report.append("- None.")
    (out / "CHECKPOINT_FRONTIER_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    manifest = {
        "phase": 11,
        "work_packages": ["C", "D"],
        "state": "complete" if not failures else "failed",
        "checkpoints": list(CHECKPOINTS),
        "tie_atol": TIE_ATOL,
        "fixed_absolute_inner_f1_tolerances": list(F1_TOLERANCES),
        "methods": list(METHODS),
        "seeds": list(range(1001, 1021)),
        "dataset_sha256": sha256_file(args.dataset),
        "checkpoint_archive_sha256": sha256_file(args.archive),
        "validation_failures": failures,
        "outer_test_used_for_selection": False,
        "hypervolume_used": False,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    sums = []
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "sha256sums.txt":
            sums.append(f"{sha256_file(path)}  {path.name}")
    (out / "sha256sums.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
