#!/usr/bin/env python3
"""Phase 11 Work Package A: audit frozen Stage-3 trace provenance without rerunning search."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import tarfile
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder


SEARCH_METHODS = (
    "random_search",
    "ga_standard",
    "ga_mutation_only",
    "ga_tournament_only",
    "ga_both",
)
EXPECTED_SEEDS = tuple(range(1001, 1021))
EXPECTED_BUDGET = 200
EXPECTED_DIMENSION = 16
PENALTY = 0.001


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_ints(values) -> str:
    payload = ",".join(str(int(v)) for v in values).encode("ascii")
    return sha256_bytes(payload)


def bit_indices(mask: int, dimension: int = EXPECTED_DIMENSION) -> list[int]:
    return [i for i in range(dimension) if mask & (1 << i)]


def safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if destination != target and destination not in target.parents:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
        tar.extractall(destination)


def git_show_bytes(commit: str, path: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def load_status_map(path: Path) -> dict[tuple[int, str], dict]:
    items = json.loads(path.read_text(encoding="utf-8"))
    return {(int(x["seed"]), x["method"]): x for x in items}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--stage3-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--dataset", type=Path, default=Path("data/Dry_Bean_Dataset.xlsx"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    warnings: list[str] = []

    execution = json.loads((args.stage3_root / "execution-summary.json").read_text(encoding="utf-8"))
    core_manifest = json.loads((args.stage3_root / "core/manifest.json").read_text(encoding="utf-8"))
    nca_manifest = json.loads((args.stage3_root / "nca/manifest.json").read_text(encoding="utf-8"))
    core_status = load_status_map(args.stage3_root / "core/unit-statuses.json")
    nca_status = load_status_map(args.stage3_root / "nca/unit-statuses.json")
    status_map = {**core_status, **nca_status}

    if execution.get("source_commit") != args.source_commit:
        failures.append(
            f"execution-summary source_commit {execution.get('source_commit')} != requested {args.source_commit}"
        )

    if core_manifest.get("state") != "complete":
        failures.append("core manifest state is not complete")
    if nca_manifest.get("state") != "complete":
        failures.append("NCA manifest state is not complete")
    if len(core_status) != 180:
        failures.append(f"expected 180 core unit statuses, found {len(core_status)}")
    if len(nca_status) != 20:
        failures.append(f"expected 20 NCA unit statuses, found {len(nca_status)}")
    if any(x.get("state") != "complete" for x in status_map.values()):
        failures.append("one or more archived unit statuses are not complete")

    expected_dataset_sha = core_manifest["identity"]["dataset_sha256"]
    actual_dataset_sha = sha256_file(args.dataset)
    if actual_dataset_sha != expected_dataset_sha:
        failures.append(
            f"dataset SHA256 mismatch: expected {expected_dataset_sha}, found {actual_dataset_sha}"
        )
    if nca_manifest["identity"]["dataset_sha256"] != expected_dataset_sha:
        failures.append("core and NCA manifests disagree on dataset SHA256")

    source_hash_rows = []
    for filename, expected_sha in core_manifest["identity"]["source_sha256"].items():
        repo_path = f"src/{filename}"
        try:
            data = git_show_bytes(args.source_commit, repo_path)
            actual_sha = sha256_bytes(data)
            ok = actual_sha == expected_sha
        except subprocess.CalledProcessError:
            actual_sha = ""
            ok = False
        source_hash_rows.append(
            {
                "file": repo_path,
                "expected_sha256": expected_sha,
                "actual_sha256": actual_sha,
                "match": ok,
            }
        )
        if not ok:
            failures.append(f"source hash mismatch or missing at {repo_path}")
    if core_manifest["identity"]["source_sha256"] != nca_manifest["identity"]["source_sha256"]:
        failures.append("core and NCA manifests disagree on frozen source hashes")

    with (out / "source_hash_audit.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(source_hash_rows[0]))
        writer.writeheader()
        writer.writerows(source_hash_rows)

    frame = pd.read_excel(args.dataset)
    if "Class" not in frame.columns:
        failures.append("dataset does not contain Class target")
        return 2
    features = frame.drop(columns="Class")
    X = features.to_numpy(dtype=float)
    encoder = LabelEncoder()
    y = encoder.fit_transform(frame["Class"])
    class_distribution = {
        str(label): int(count)
        for label, count in frame["Class"].value_counts().sort_index().items()
    }
    if X.shape != (13611, 16):
        failures.append(f"unexpected dataset feature shape {X.shape}")
    if len(np.unique(y)) != 7:
        failures.append(f"unexpected class count {len(np.unique(y))}")

    outer_expected: dict[int, tuple[list[int], list[int]]] = {}
    fold_rows: list[dict] = []
    for seed in EXPECTED_SEEDS:
        train, test = train_test_split(
            np.arange(len(y)),
            test_size=0.2,
            stratify=y,
            random_state=seed,
        )
        outer_expected[seed] = (train.tolist(), test.tolist())
        y_train = y[train]
        splitter = StratifiedKFold(3, shuffle=True, random_state=seed)
        for fold_id, (inner_train, inner_valid) in enumerate(
            splitter.split(X[train], y_train), start=1
        ):
            train_rows = train[inner_train].tolist()
            valid_rows = train[inner_valid].tolist()
            fold_rows.append(
                {
                    "seed": seed,
                    "fold": fold_id,
                    "inner_train_count": len(train_rows),
                    "inner_valid_count": len(valid_rows),
                    "inner_train_outer_row_sha256": digest_ints(train_rows),
                    "inner_valid_outer_row_sha256": digest_ints(valid_rows),
                }
            )

    with (out / "inner_fold_inventory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fold_rows[0]))
        writer.writeheader()
        writer.writerows(fold_rows)

    inventory: list[dict] = []
    first20_by_seed: dict[int, dict[str, list[int]]] = defaultdict(dict)
    outer_by_seed: dict[int, dict[str, tuple[str, str]]] = defaultdict(dict)
    search_units = 0
    all_units = 0

    with tempfile.TemporaryDirectory(prefix="phase11-trace-audit-") as tmp:
        extracted = Path(tmp)
        safe_extract(args.archive, extracted)
        result_paths = sorted(extracted.rglob("result.json"))
        all_units = len(result_paths)
        if all_units != 200:
            failures.append(f"checkpoint archive contains {all_units} result.json files, expected 200")

        seen_units: set[tuple[int, str]] = set()
        for result_path in result_paths:
            raw = result_path.read_bytes()
            result_sha = sha256_bytes(raw)
            result = json.loads(raw)
            seed = int(result["seed"])
            method = result["method"]
            unit = (seed, method)
            if unit in seen_units:
                failures.append(f"duplicate unit in archive: {seed}-{method}")
            seen_units.add(unit)

            status = status_map.get(unit)
            status_sha = status.get("result_sha256") if status else None
            status_hash_match = result_sha == status_sha
            if not status:
                failures.append(f"archive unit missing from unit-statuses: {seed}-{method}")
            elif not status_hash_match:
                failures.append(f"result SHA mismatch for {seed}-{method}")

            train_indices = [int(v) for v in result.get("train_indices", [])]
            test_indices = [int(v) for v in result.get("test_indices", [])]
            expected_train, expected_test = outer_expected.get(seed, ([], []))
            outer_reproduces = train_indices == expected_train and test_indices == expected_test
            if not outer_reproduces:
                failures.append(f"outer split mismatch for {seed}-{method}")

            train_digest = digest_ints(train_indices)
            test_digest = digest_ints(test_indices)
            outer_by_seed[seed][method] = (train_digest, test_digest)

            y_true = result.get("y_true", [])
            if len(y_true) != len(test_indices):
                failures.append(f"y_true length mismatch for {seed}-{method}")
            for classifier, metrics in result.get("classifiers", {}).items():
                if len(metrics.get("predictions", [])) != len(test_indices):
                    failures.append(f"prediction length mismatch for {seed}-{method}-{classifier}")

            search = result.get("search")
            is_search = method in SEARCH_METHODS
            history_len = None
            unique_masks = None
            initial20_digest = None
            final_mask = None
            formula_ok = None
            cv_mean_ok = None
            final_traceable = None
            eval_order_ok = None
            valid_masks_ok = None
            selected_mapping_ok = None
            archived_unique_evaluations = None
            archived_cv_model_fits = None

            if is_search:
                search_units += 1
                if not isinstance(search, dict):
                    failures.append(f"missing search object for {seed}-{method}")
                    history = []
                else:
                    history = search.get("history", [])
                history_len = len(history)
                masks = [int(row["mask"]) for row in history]
                unique_masks = len(set(masks))
                evals = [int(row.get("evaluation", -1)) for row in history]
                eval_order_ok = evals == list(range(1, len(history) + 1))
                valid_masks_ok = all(1 <= m < (1 << EXPECTED_DIMENSION) for m in masks)
                formula_ok = True
                cv_mean_ok = True
                for row in history:
                    selected_count = int(row["selected_count"])
                    mask = int(row["mask"])
                    if selected_count != len(bit_indices(mask)):
                        formula_ok = False
                    folds = [float(v) for v in row["fold_scores"]]
                    cv_mean = float(row["cv_macro_f1"])
                    if len(folds) != 3 or not math.isclose(
                        statistics.fmean(folds), cv_mean, rel_tol=0.0, abs_tol=args.tolerance
                    ):
                        cv_mean_ok = False
                    expected_fitness = cv_mean - PENALTY * selected_count / EXPECTED_DIMENSION
                    if not math.isclose(
                        float(row["fitness"]),
                        expected_fitness,
                        rel_tol=0.0,
                        abs_tol=args.tolerance,
                    ):
                        formula_ok = False

                archived_unique_evaluations = int(search.get("unique_evaluations", -1))
                archived_cv_model_fits = int(search.get("cv_model_fits", -1))
                if history_len != EXPECTED_BUDGET:
                    failures.append(f"{seed}-{method}: history length {history_len}, expected 200")
                if unique_masks != EXPECTED_BUDGET:
                    failures.append(f"{seed}-{method}: unique mask count {unique_masks}, expected 200")
                if archived_unique_evaluations != EXPECTED_BUDGET:
                    failures.append(
                        f"{seed}-{method}: archived unique_evaluations "
                        f"{archived_unique_evaluations}, expected 200"
                    )
                if archived_cv_model_fits != 3 * EXPECTED_BUDGET:
                    failures.append(
                        f"{seed}-{method}: archived cv_model_fits {archived_cv_model_fits}, expected 600"
                    )
                if not eval_order_ok:
                    failures.append(f"{seed}-{method}: evaluation order is not 1..200")
                if not valid_masks_ok:
                    failures.append(f"{seed}-{method}: invalid/zero subset mask found")
                if not formula_ok:
                    failures.append(f"{seed}-{method}: fitness/cardinality formula mismatch")
                if not cv_mean_ok:
                    failures.append(f"{seed}-{method}: cv_macro_f1 does not equal mean fold score")

                first20 = masks[:20]
                first20_by_seed[seed][method] = first20
                initial20_digest = digest_ints(first20)

                best = search.get("best", {})
                final_mask = int(best.get("mask", -1))
                row_by_mask = {int(row["mask"]): row for row in history}
                final_traceable = final_mask in row_by_mask
                if not final_traceable:
                    failures.append(f"{seed}-{method}: final best mask not found in trace")

                def stage3_key(row):
                    return (
                        float(row["fitness"]),
                        -int(row["selected_count"]),
                        -int(row["mask"]),
                    )

                if history:
                    deterministic_best = max(history, key=stage3_key)
                    if final_mask != int(deterministic_best["mask"]):
                        failures.append(f"{seed}-{method}: archived best violates frozen Stage-3 tie rule")
                selected_mapping_ok = result.get("selected_original_indices") == bit_indices(final_mask)
                if not selected_mapping_ok:
                    failures.append(f"{seed}-{method}: selected indices do not match final mask")

            elif search is not None:
                warnings.append(f"non-search method {seed}-{method} unexpectedly has a search object")

            inventory.append(
                {
                    "seed": seed,
                    "method": method,
                    "archive_member": str(result_path.relative_to(extracted)),
                    "result_sha256": result_sha,
                    "status_result_sha256": status_sha or "",
                    "status_hash_match": status_hash_match,
                    "outer_split_reproduces": outer_reproduces,
                    "train_indices_sha256": train_digest,
                    "test_indices_sha256": test_digest,
                    "is_search_method": is_search,
                    "history_len": history_len if history_len is not None else "",
                    "unique_masks": unique_masks if unique_masks is not None else "",
                    "archived_unique_evaluations": (
                        archived_unique_evaluations
                        if archived_unique_evaluations is not None
                        else ""
                    ),
                    "archived_cv_model_fits": (
                        archived_cv_model_fits if archived_cv_model_fits is not None else ""
                    ),
                    "evaluation_order_ok": eval_order_ok if eval_order_ok is not None else "",
                    "valid_nonempty_masks_ok": valid_masks_ok if valid_masks_ok is not None else "",
                    "fitness_formula_ok": formula_ok if formula_ok is not None else "",
                    "cv_mean_from_fold_scores_ok": cv_mean_ok if cv_mean_ok is not None else "",
                    "initial20_sha256": initial20_digest or "",
                    "final_mask": final_mask if final_mask is not None else "",
                    "final_mask_traceable": final_traceable if final_traceable is not None else "",
                    "selected_mapping_ok": selected_mapping_ok if selected_mapping_ok is not None else "",
                }
            )

        missing_status_units = sorted(set(status_map) - seen_units)
        if missing_status_units:
            failures.append(
                "checkpoint archive is missing status units: "
                + ", ".join(f"{s}-{m}" for s, m in missing_status_units[:20])
            )

    for seed in EXPECTED_SEEDS:
        traces = first20_by_seed.get(seed, {})
        if set(traces) != set(SEARCH_METHODS):
            failures.append(
                f"seed {seed}: expected initial traces for all five search methods, found {sorted(traces)}"
            )
            continue
        initial_sequences = list(traces.values())
        if any(seq != initial_sequences[0] for seq in initial_sequences[1:]):
            failures.append(f"seed {seed}: first 20 masks differ across search methods")

        split_digests = set(outer_by_seed.get(seed, {}).values())
        if len(split_digests) != 1:
            failures.append(f"seed {seed}: outer split indices differ across methods")

    if search_units != 100:
        failures.append(f"found {search_units} search units, expected 100")

    inventory.sort(key=lambda r: (r["seed"], r["method"]))
    with (out / "trace_inventory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(inventory[0]))
        writer.writeheader()
        writer.writerows(inventory)

    method_counts = Counter(r["method"] for r in inventory)
    all_initial20_equal = all(
        len({digest_ints(seq) for seq in first20_by_seed.get(seed, {}).values()}) == 1
        and len(first20_by_seed.get(seed, {})) == len(SEARCH_METHODS)
        for seed in EXPECTED_SEEDS
    )
    all_outer_equal = all(
        len(set(outer_by_seed.get(seed, {}).values())) == 1
        and len(outer_by_seed.get(seed, {})) == 10
        for seed in EXPECTED_SEEDS
    )

    summary = {
        "phase": 11,
        "work_package": "A",
        "gate_a_passed": not failures,
        "archive": {
            "path": str(args.archive),
            "sha256": sha256_file(args.archive),
            "result_units": all_units,
            "search_units": search_units,
            "method_counts": dict(sorted(method_counts.items())),
        },
        "dataset_profile": {
            "rows": int(X.shape[0]),
            "input_features": int(X.shape[1]),
            "classes": int(len(np.unique(y))),
            "class_distribution": class_distribution,
            "feature_names": features.columns.tolist(),
        },
        "frozen_identity": {
            "source_commit_expected": args.source_commit,
            "source_commit_execution_summary": execution.get("source_commit"),
            "dataset_sha256_expected": expected_dataset_sha,
            "dataset_sha256_actual": actual_dataset_sha,
            "python_archived": core_manifest["identity"].get("python"),
            "platform_archived": core_manifest["identity"].get("platform"),
            "packages_archived": core_manifest["identity"].get("packages"),
            "thread_environment_archived": core_manifest["identity"].get("thread_environment"),
        },
        "checks": {
            "all_unit_statuses_complete": all(
                x.get("state") == "complete" for x in status_map.values()
            )
            and len(core_status) == 180
            and len(nca_status) == 20,
            "all_result_hashes_match_statuses": all(bool(r["status_hash_match"]) for r in inventory),
            "all_outer_splits_reproduce_from_dataset_seed": all(
                bool(r["outer_split_reproduces"]) for r in inventory
            ),
            "same_outer_split_across_methods_within_seed": all_outer_equal,
            "all_search_histories_have_200_unique_nonempty_masks": all(
                r["history_len"] == 200
                and r["unique_masks"] == 200
                and r["valid_nonempty_masks_ok"] is True
                for r in inventory
                if r["is_search_method"]
            ),
            "deterministic_evaluation_order_recoverable": all(
                r["evaluation_order_ok"] is True
                for r in inventory
                if r["is_search_method"]
            ),
            "same_initial_20_masks_across_search_methods_within_seed": all_initial20_equal,
            "fitness_formula_reproduces": all(
                r["fitness_formula_ok"] is True
                for r in inventory
                if r["is_search_method"]
            ),
            "cv_mean_matches_archived_fold_scores": all(
                r["cv_mean_from_fold_scores_ok"] is True
                for r in inventory
                if r["is_search_method"]
            ),
            "final_masks_traceable": all(
                r["final_mask_traceable"] is True and r["selected_mapping_ok"] is True
                for r in inventory
                if r["is_search_method"]
            ),
            "source_hashes_match_manifest": all(r["match"] for r in source_hash_rows),
            "dataset_hash_matches_manifest": actual_dataset_sha == expected_dataset_sha,
            "inner_folds_reconstructed_from_frozen_seed_and_outer_training_order": True,
        },
        "inner_fold_provenance_note": (
            "Stage-3 result files archive per-candidate fold scores but not inner fold row indices. "
            "Exact inner folds are deterministically recoverable from the verified dataset, outer-training "
            "order, seed, frozen scikit-learn protocol, and source code. inner_fold_inventory.csv records "
            "reconstructed original-row digests for all 20 seeds."
        ),
        "static_source_audit": {
            "search_receives_outer_training_only": True,
            "scaling_is_inside_inner_cv_pipeline": True,
            "outer_test_not_referenced_by_budgeted_search": True,
            "basis": (
                "Verified frozen source hashes plus source inspection: stage2.py calls "
                "budgeted_search(X_train, y_train, ...); budgeted_search.py constructs "
                "StratifiedKFold(3, shuffle=True, random_state=seed) and evaluates "
                "make_pipeline(StandardScaler(), LogisticRegression(...)) via cross_val_score."
            ),
        },
        "failures": failures,
        "warnings": warnings,
    }
    (out / "trace_audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    md = [
        "# Phase 11 Work Package A — Trace Provenance Audit",
        "",
        f"- Gate A: **{'PASS' if not failures else 'NOT PASSED'}**",
        f"- Archived result units inspected: {all_units}",
        f"- Search units inspected: {search_units}",
        f"- Checkpoint archive SHA256: {summary['archive']['sha256']}",
        f"- Frozen source commit: {args.source_commit}",
        f"- Dataset SHA256 match: {actual_dataset_sha == expected_dataset_sha}",
        f"- First 20 masks identical across all five search methods within every seed: {all_initial20_equal}",
        f"- Outer split identical across all methods within every seed: {all_outer_equal}",
        "",
        "## Important provenance note",
        "",
        summary["inner_fold_provenance_note"],
        "",
        "## Failures",
        "",
    ]
    if failures:
        md.extend(f"- {item}" for item in failures)
    else:
        md.append("- None.")
    md += ["", "## Warnings", ""]
    if warnings:
        md.extend(f"- {item}" for item in warnings)
    else:
        md.append("- None.")
    md += [
        "",
        "## Generated evidence",
        "",
        "- trace_audit_summary.json",
        "- trace_inventory.csv",
        "- inner_fold_inventory.csv",
        "- source_hash_audit.csv",
        "",
    ]
    (out / "TRACE_AUDIT_REPORT.md").write_text("\n".join(md), encoding="utf-8")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
