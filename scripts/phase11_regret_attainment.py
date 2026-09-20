#!/usr/bin/env python3
"""Phase 11 Work Package F: exact regret, attainment timing, and exact-optimum outer transfer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import tarfile
import tempfile
import time
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

METHODS = tuple(SEARCH_METHODS)
BUDGETS = (20, 50, 100, 150, 200)
EXACT_ATOL = 1e-12
NEAR_TOLS = (0.001, 0.005)


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


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    if fields is None:
        fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_archived_units(root: Path) -> dict[tuple[int, str], dict]:
    units = {}
    for path in root.rglob("result.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        method = data.get("method")
        seed = data.get("seed")
        if method in METHODS and seed is not None:
            key = (int(seed), str(method))
            if key in units:
                raise RuntimeError(f"Duplicate archived unit {key}")
            units[key] = data
    expected = {(s, m) for s in range(1001, 1021) for m in METHODS}
    missing = sorted(expected - set(units))
    if missing:
        raise RuntimeError(f"Missing archived units: {missing[:5]}")
    return units


def load_dataset(dataset: Path, seed: int):
    frame = pd.read_excel(dataset)
    features = frame.drop(columns="Class")
    X = features.to_numpy(dtype=float)
    y = LabelEncoder().fit_transform(frame["Class"])
    train, test = train_test_split(
        np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed
    )
    return features, X, y, train, test


def evaluate_exact_mask(dataset: Path, seed: int, mask: int):
    features, X, y, train, test = load_dataset(dataset, seed)
    selected = mask_indices(mask, X.shape[1])
    X_train, X_test = X[train][:, selected], X[test][:, selected]
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
            "seed": seed,
            "mask": mask,
            "mask_bits": format(mask, "016b"),
            "selected_count": len(selected),
            "feature_names": "|".join(features.columns[selected].tolist()),
            "classifier": name,
            "macro_f1": float(f1_score(y_test, pred, average="macro")),
            "accuracy": float(accuracy_score(y_test, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
            "fit_seconds": fit_s,
            "predict_seconds": pred_s,
        })
        predictions[name] = np.asarray(pred, dtype=np.int16)
    return rows, predictions, test, y_test


def describe(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--checkpoint-results", type=Path, required=True)
    p.add_argument("--exact-optima", type=Path, required=True)
    p.add_argument("--exact-cardinality", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    cp_all = read_csv(args.checkpoint_results)
    cp = [r for r in cp_all if r["classifier"] == "Logistic Regression"]
    expected_cp = 20 * len(METHODS) * len(BUDGETS)
    if len(cp) != expected_cp:
        raise RuntimeError(f"Expected {expected_cp} LR checkpoint rows, got {len(cp)}")

    exact_rows = read_csv(args.exact_optima)
    if len(exact_rows) != 20:
        raise RuntimeError("Expected 20 exact optimum rows")
    exact = {int(r["seed"]): r for r in exact_rows}

    card_rows = read_csv(args.exact_cardinality)
    if len(card_rows) != 20 * 16:
        raise RuntimeError("Expected 320 exact cardinality rows")
    card = {(int(r["seed"]), int(r["selected_count"])): r for r in card_rows}

    with tempfile.TemporaryDirectory(prefix="phase11-f-") as tmp:
        root = Path(tmp)
        safe_extract(args.archive, root)
        archived = load_archived_units(root)

        regret_rows = []
        timing_rows = []
        validation_failures = []

        cp_lookup = {
            (int(r["seed"]), r["method"], int(r["budget"])): r for r in cp
        }

        for seed in range(1001, 1021):
            exact_fit = float(exact[seed]["fitness"])
            exact_mask = int(exact[seed]["mask"])
            exact_count = int(exact[seed]["selected_count"])

            for method in METHODS:
                history = archived[(seed, method)]["search"]["history"]
                if len(history) != 200:
                    validation_failures.append(f"{seed}-{method}: history length != 200")
                    continue

                final = cp_lookup[(seed, method, 200)]
                final_mask = int(final["mask"])
                positions = [i + 1 for i, r in enumerate(history) if int(r["mask"]) == final_mask]
                if len(positions) != 1:
                    validation_failures.append(f"{seed}-{method}: final mask occurrence != 1")
                    final_eval = None
                else:
                    final_eval = positions[0]

                cumulative_best = -math.inf
                first_near = {tol: None for tol in NEAR_TOLS}
                for i, r in enumerate(history, start=1):
                    cumulative_best = max(cumulative_best, float(r["fitness"]))
                    regret = exact_fit - cumulative_best
                    for tol in NEAR_TOLS:
                        if first_near[tol] is None and regret <= tol + EXACT_ATOL:
                            first_near[tol] = i

                timing_rows.append({
                    "seed": seed,
                    "method": method,
                    "evaluation_first_reaching_final_200_solution": final_eval if final_eval is not None else "",
                    "fraction_of_budget_first_reaching_final_200_solution": (final_eval / 200.0) if final_eval is not None else "",
                    "evaluation_first_within_0_001_exact_objective": first_near[0.001] if first_near[0.001] is not None else "",
                    "fraction_first_within_0_001": (first_near[0.001] / 200.0) if first_near[0.001] is not None else "",
                    "evaluation_first_within_0_005_exact_objective": first_near[0.005] if first_near[0.005] is not None else "",
                    "fraction_first_within_0_005": (first_near[0.005] / 200.0) if first_near[0.005] is not None else "",
                })

                for budget in BUDGETS:
                    r = cp_lookup[(seed, method, budget)]
                    selected_count = int(r["selected_count"])
                    exact_card = card[(seed, selected_count)]
                    objective_regret = exact_fit - float(r["fitness"])
                    matched_cv_regret = float(exact_card["cv_macro_f1"]) - float(r["inner_cv_macro_f1"])
                    if objective_regret < -EXACT_ATOL:
                        validation_failures.append(
                            f"{seed}-{method}-B{budget}: negative exact objective regret {objective_regret}"
                        )
                    regret_rows.append({
                        "seed": seed,
                        "method": method,
                        "budget": budget,
                        "checkpoint_mask": int(r["mask"]),
                        "checkpoint_selected_count": selected_count,
                        "checkpoint_fitness": float(r["fitness"]),
                        "checkpoint_inner_cv_macro_f1": float(r["inner_cv_macro_f1"]),
                        "exact_mask": exact_mask,
                        "exact_selected_count": exact_count,
                        "exact_fitness": exact_fit,
                        "objective_regret": max(0.0, objective_regret) if objective_regret > -EXACT_ATOL else objective_regret,
                        "matched_cardinality_exact_best_cv_macro_f1": float(exact_card["cv_macro_f1"]),
                        "matched_cardinality_inner_macro_f1_regret": max(0.0, matched_cv_regret) if matched_cv_regret > -EXACT_ATOL else matched_cv_regret,
                        "feature_count_difference_from_exact": selected_count - exact_count,
                        "exact_optimum_attained": objective_regret <= EXACT_ATOL,
                        "within_0_001_exact_objective": objective_regret <= 0.001 + EXACT_ATOL,
                        "within_0_005_exact_objective": objective_regret <= 0.005 + EXACT_ATOL,
                    })

    summary_rows = []
    for method in METHODS:
        for budget in BUDGETS:
            rows = [r for r in regret_rows if r["method"] == method and int(r["budget"]) == budget]
            vals = [float(r["objective_regret"]) for r in rows]
            matched = [float(r["matched_cardinality_inner_macro_f1_regret"]) for r in rows]
            diffs = [int(r["feature_count_difference_from_exact"]) for r in rows]
            d = describe(vals)
            dm = describe(matched)
            summary_rows.append({
                "method": method,
                "budget": budget,
                "mean_objective_regret": d["mean"],
                "median_objective_regret": d["median"],
                "min_objective_regret": d["min"],
                "max_objective_regret": d["max"],
                "mean_matched_cardinality_inner_macro_f1_regret": dm["mean"],
                "median_matched_cardinality_inner_macro_f1_regret": dm["median"],
                "mean_feature_count_difference_from_exact": float(np.mean(diffs)),
                "exact_optimum_attainment_count": sum(bool(r["exact_optimum_attained"]) for r in rows),
                "within_0_001_count": sum(bool(r["within_0_001_exact_objective"]) for r in rows),
                "within_0_005_count": sum(bool(r["within_0_005_exact_objective"]) for r in rows),
            })

    paired_rows = []
    random_lookup = {
        (int(r["seed"]), int(r["budget"])): float(r["objective_regret"])
        for r in regret_rows if r["method"] == "random_search"
    }
    for method in METHODS:
        if method == "random_search":
            continue
        for budget in BUDGETS:
            pairs = []
            for r in regret_rows:
                if r["method"] == method and int(r["budget"]) == budget:
                    seed = int(r["seed"])
                    diff = float(r["objective_regret"]) - random_lookup[(seed, budget)]
                    pairs.append(diff)
            d = describe(pairs)
            paired_rows.append({
                "ga_method": method,
                "budget": budget,
                "mean_paired_regret_difference_ga_minus_random": d["mean"],
                "median_paired_regret_difference_ga_minus_random": d["median"],
                "min_paired_difference": d["min"],
                "max_paired_difference": d["max"],
                "ga_lower_regret_wins": sum(x < -EXACT_ATOL for x in pairs),
                "ties": sum(abs(x) <= EXACT_ATOL for x in pairs),
                "random_lower_regret_wins": sum(x > EXACT_ATOL for x in pairs),
            })

    exact_outer_rows = []
    predictions = {}
    exact_outer_by_key = {}
    for seed in range(1001, 1021):
        mask = int(exact[seed]["mask"])
        rows, preds, test_idx, y_true = evaluate_exact_mask(args.dataset, seed, mask)
        exact_outer_rows.extend(rows)
        predictions[f"s{seed}_test_indices"] = np.asarray(test_idx, dtype=np.int32)
        predictions[f"s{seed}_y_true"] = np.asarray(y_true, dtype=np.int16)
        for row in rows:
            cname = row["classifier"]
            predictions[f"s{seed}_exact_{cname.lower().replace(' ', '_')}"] = preds[cname]
            exact_outer_by_key[(seed, cname)] = row

    cp200 = [r for r in cp_all if int(r["budget"]) == 200]
    transfer_compare = []
    for r in cp200:
        seed = int(r["seed"])
        ex = exact_outer_by_key[(seed, r["classifier"])]
        transfer_compare.append({
            "seed": seed,
            "method": r["method"],
            "classifier": r["classifier"],
            "budget_200_selected_count": int(r["selected_count"]),
            "exact_selected_count": int(ex["selected_count"]),
            "budget_200_outer_macro_f1": float(r["macro_f1"]),
            "exact_optimum_outer_macro_f1": float(ex["macro_f1"]),
            "exact_minus_budget_200_outer_macro_f1": float(ex["macro_f1"]) - float(r["macro_f1"]),
            "budget_200_outer_accuracy": float(r["accuracy"]),
            "exact_optimum_outer_accuracy": float(ex["accuracy"]),
            "exact_minus_budget_200_outer_accuracy": float(ex["accuracy"]) - float(r["accuracy"]),
        })

    write_csv(out / "regret_by_seed_method_budget.csv", regret_rows)
    write_csv(out / "regret_summary.csv", summary_rows)
    write_csv(out / "ga_vs_random_paired_regret.csv", paired_rows)
    write_csv(out / "attainment_timing.csv", timing_rows)
    write_csv(out / "exact_optimum_outer_results.csv", exact_outer_rows)
    write_csv(out / "exact_vs_budget200_outer_transfer.csv", transfer_compare)
    np.savez_compressed(out / "exact_optimum_outer_predictions.npz", **predictions)

    report = [
        "# Phase 11 Work Package F — Regret and Attainment",
        "",
        f"- Validation: **{'PASS' if not validation_failures else 'FAIL'}**",
        "- Exact-optimum attainment tolerance: 1e-12 absolute frozen objective regret.",
        "- Descriptive near-optimum regret thresholds: 0.001 and 0.005, inherited from the previously frozen Phase-11 score tolerances.",
        "- All regret calculations are paired within the same frozen outer split.",
        "- Repeated holdouts are dependent; no independent-dataset significance interpretation is made.",
        "- Exact-objective optima were evaluated on untouched outer tests only after exact training-only selection.",
        "- Exact reference is not treated as a 200-evaluation competitor.",
        "",
        "## Validation failures",
        "",
    ]
    report.extend(f"- {x}" for x in validation_failures) if validation_failures else report.append("- None.")
    (out / "REGRET_ATTAINMENT_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    manifest = {
        "phase": 11,
        "work_package": "F",
        "state": "complete" if not validation_failures else "failed",
        "exact_attainment_atol": EXACT_ATOL,
        "near_optimum_regret_thresholds": list(NEAR_TOLS),
        "seeds": list(range(1001, 1021)),
        "methods": list(METHODS),
        "budgets": list(BUDGETS),
        "dataset_sha256": sha256_file(args.dataset),
        "checkpoint_archive_sha256": sha256_file(args.archive),
        "checkpoint_results_sha256": sha256_file(args.checkpoint_results),
        "exact_optima_sha256": sha256_file(args.exact_optima),
        "exact_cardinality_sha256": sha256_file(args.exact_cardinality),
        "outer_test_used_for_exact_selection": False,
        "validation_failures": validation_failures,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    sums = []
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "sha256sums.txt":
            sums.append(f"{sha256_file(path)}  {path.name}")
    (out / "sha256sums.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")

    return 0 if not validation_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
