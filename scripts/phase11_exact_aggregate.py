#!/usr/bin/env python3
"""Aggregate compact Phase-11 exact-reference summaries from per-seed workflow artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--run-attempt", default="1")
    args = p.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    optima = []
    cardinality = []
    index = []

    for seed in range(1001, 1021):
        matches = list(args.artifacts_root.rglob(f"phase11-exact-seed-{seed}"))
        root = matches[0] if matches else None
        if root is None:
            # download-artifact may merge artifact contents directly under a directory.
            candidates = [p.parent for p in args.artifacts_root.rglob("manifest.json")
                          if json.loads(p.read_text()).get("seed") == seed]
            root = candidates[0] if candidates else None
        if root is None:
            raise FileNotFoundError(f"artifact for seed {seed} not found")

        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("state") != "complete" or int(manifest.get("nonempty_masks", 0)) != 65535:
            raise RuntimeError(f"seed {seed} is not a complete exact split")

        o = read_csv(root / "exact_optimum.csv")
        if len(o) != 1:
            raise RuntimeError(f"seed {seed} optimum row count != 1")
        optima.extend(o)

        c = read_csv(root / "best_by_cardinality.csv")
        if len(c) != 16:
            raise RuntimeError(f"seed {seed} cardinality rows != 16")
        cardinality.extend(c)

        index.append({
            "seed": seed,
            "artifact_name": f"phase11-exact-seed-{seed}",
            "artifact_manifest_sha256": sha256_file(manifest_path),
            "scientific_sha256sums_sha256": sha256_file(root / "sha256sums.txt"),
            "elapsed_wall_seconds": manifest["elapsed_wall_seconds"],
            "warnings_by_mask_category": json.dumps(manifest.get("warnings_by_mask_category", {}), sort_keys=True),
            "dataset_sha256": manifest["dataset_sha256"],
            "budgeted_search_sha256": manifest["budgeted_search_sha256"],
            "outer_train_indices_sha256": manifest["outer_train_indices_sha256"],
            "outer_test_indices_sha256": manifest["outer_test_indices_sha256"],
        })

    if [int(r["seed"]) for r in optima] != list(range(1001, 1021)):
        raise RuntimeError("optimum seed coverage mismatch")
    if len(cardinality) != 20 * 16:
        raise RuntimeError("cardinality summary size mismatch")
    if len({r["dataset_sha256"] for r in index}) != 1:
        raise RuntimeError("dataset hash differs across exact split artifacts")
    if len({r["budgeted_search_sha256"] for r in index}) != 1:
        raise RuntimeError("scoring source hash differs across exact split artifacts")

    write_csv(out / "exact_optima.csv", optima,
              ["seed", "mask", "mask_bits", "selected_count", "cv_macro_f1", "fitness"])
    write_csv(out / "best_by_cardinality_all.csv", cardinality,
              ["seed", "selected_count", "mask", "mask_bits", "cv_macro_f1", "fitness"])
    write_csv(out / "artifact_index.csv", index, list(index[0]))

    meta = {
        "phase": 11,
        "work_package": "E",
        "state": "complete",
        "exact_splits": 20,
        "masks_per_split": 65535,
        "total_subset_objectives": 20 * 65535,
        "total_inner_model_fits": 20 * 65535 * 3,
        "github_actions_run_id": str(args.run_id),
        "github_actions_run_attempt": str(args.run_attempt),
        "storage_policy": "Per-mask archives retained as GitHub Actions artifacts; repository stores compact summaries, manifests/hashes only.",
        "dataset_sha256": index[0]["dataset_sha256"],
        "budgeted_search_sha256": index[0]["budgeted_search_sha256"],
    }
    (out / "EXACT_REFERENCE_INDEX.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = [
        "# Phase 11 Work Package E — Exact Reference Enumeration",
        "",
        "- Status: **COMPLETE**",
        "- Exact outer-training splits: 20/20",
        "- Nonempty masks per split: 65,535",
        f"- Total subset objectives: {20 * 65535:,}",
        f"- Total inner logistic-regression fits: {20 * 65535 * 3:,}",
        f"- GitHub Actions run: {args.run_id} (attempt {args.run_attempt})",
        "- Outer test data used during exact search: **NO**",
        "- Heavy per-mask archives: retained as per-seed GitHub Actions artifacts.",
        "- Repository: compact exact optima, cardinality summaries, artifact hashes and provenance only.",
        "",
        "Exact-objective optima must not be described as equal-budget competitors. Their role is to provide an attainable-objective reference for the frozen training-only search objective.",
        "",
    ]
    (out / "EXACT_REFERENCE_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
