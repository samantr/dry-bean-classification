from pathlib import Path
from itertools import combinations
import argparse

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_results_dir() -> Path:
    return get_project_root() / "results" / "comparison"


def load_detailed_results(results_dir: Path) -> pd.DataFrame:
    file_path = results_dir / "detailed_results_long.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Could not find: {file_path}")

    df = pd.read_csv(file_path)

    required_cols = {"seed", "method", "classifier", "macro_f1"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in detailed_results_long.csv: {missing}")
    validate_results(df)

    return df


def validate_results(df):
    if df.duplicated(['seed', 'method', 'classifier']).any():
        raise ValueError('Duplicate seed/method/classifier rows; cannot silently average repeated runs')
    if not np.isfinite(df['macro_f1']).all() or not df['macro_f1'].between(0, 1).all():
        raise ValueError('macro_f1 must be finite and in [0, 1]')


def holm_adjust(p_values):
    """Holm family-wise correction for all planned finite pairwise tests."""
    p = np.asarray(p_values, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Expected finite p-values in [0, 1]')
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted


def run_friedman_by_classifier(df: pd.DataFrame) -> pd.DataFrame:
    validate_results(df)
    rows = []

    classifiers = sorted(df["classifier"].unique())

    for clf in classifiers:
        sub = df[df["classifier"] == clf].copy()

        pivot = sub.pivot_table(
            index="seed",
            columns="method",
            values="macro_f1",
            aggfunc="mean"
        )

        required_methods = ["baseline", "vanilla_ga", "improved_ga", "nca"]
        missing_methods = [m for m in required_methods if m not in pivot.columns]
        if missing_methods:
            print(f"Skipping Friedman for {clf}; missing methods: {missing_methods}")
            continue

        pivot = pivot[required_methods].dropna()

        if len(pivot) < 2:
            print(f"Skipping Friedman for {clf}; not enough paired seeds.")
            continue

        stat, p_value = friedmanchisquare(
            pivot["baseline"],
            pivot["vanilla_ga"],
            pivot["improved_ga"],
            pivot["nca"]
        )

        rows.append({
            "classifier": clf,
            "num_seeds": len(pivot),
            "friedman_statistic": stat,
            "friedman_p_value": p_value,
            "significant_at_0_05": p_value < 0.05
        })

    return pd.DataFrame(rows)


def run_pairwise_wilcoxon(df: pd.DataFrame) -> pd.DataFrame:
    validate_results(df)
    rows = []

    classifiers = sorted(df["classifier"].unique())

    for clf in classifiers:
        sub = df[df["classifier"] == clf].copy()

        pivot = sub.pivot_table(
            index="seed",
            columns="method",
            values="macro_f1",
            aggfunc="mean"
        )

        methods = [m for m in ["baseline", "vanilla_ga", "improved_ga", "nca"] if m in pivot.columns]

        for m1, m2 in combinations(methods, 2):
            paired = pivot[[m1, m2]].dropna()

            if len(paired) < 2:
                continue

            diffs = paired[m1] - paired[m2]
            all_zero = (diffs.abs() < 1e-15).all()

            if all_zero:
                stat = 0.0
                p_value = 1.0
            else:
                try:
                    stat, p_value = wilcoxon(
                        paired[m1],
                        paired[m2],
                        zero_method="wilcox",
                        alternative="two-sided"
                    )
                except ValueError:
                    # fallback for edge cases with tiny paired samples
                    stat = None
                    p_value = None

            rows.append({
                "classifier": clf,
                "method_1": m1,
                "method_2": m2,
                "num_seeds": len(paired),
                "wilcoxon_statistic": stat,
                "wilcoxon_p_value": p_value,
                "significant_at_0_05": (p_value is not None and p_value < 0.05),
                "mean_macro_f1_method_1": paired[m1].mean(),
                "mean_macro_f1_method_2": paired[m2].mean(),
                "mean_difference_method1_minus_method2": (paired[m1] - paired[m2]).mean()
            })

    result = pd.DataFrame(rows)
    if not result.empty:
        # One correction family across classifiers and method pairs, not a
        # selectively chosen subset of comparisons. Retain raw p-values too.
        valid = result['wilcoxon_p_value'].notna()
        result['holm_p_value'] = np.nan
        result.loc[valid, 'holm_p_value'] = holm_adjust(result.loc[valid, 'wilcoxon_p_value'])
        result['significant_at_0_05'] = result['holm_p_value'] < 0.05
        result['inference_scope'] = 'exploratory_repeated_holdout_not_independent_datasets'
    return result


def build_mean_std_table(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["classifier", "method"])["macro_f1"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .sort_values(["classifier", "mean"], ascending=[True, False])
    )
    return summary


def save_dataframe(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def print_pretty_summary(friedman_df: pd.DataFrame, wilcoxon_df: pd.DataFrame, summary_df: pd.DataFrame):
    print("\n" + "=" * 90)
    print("MEAN ± STD MACRO-F1 SUMMARY")
    print("=" * 90)
    for clf in sorted(summary_df["classifier"].unique()):
        print(f"\n{clf}")
        sub = summary_df[summary_df["classifier"] == clf]
        for _, row in sub.iterrows():
            print(
                f"  {row['method']:<12} mean={row['mean']:.4f} "
                f"std={row['std']:.4f} min={row['min']:.4f} max={row['max']:.4f}"
            )

    print("\n" + "=" * 90)
    print("FRIEDMAN TEST RESULTS")
    print("=" * 90)
    if friedman_df.empty:
        print("No Friedman results.")
    else:
        print(friedman_df.to_string(index=False))

    print("\n" + "=" * 90)
    print("PAIRWISE WILCOXON RESULTS")
    print("=" * 90)
    if wilcoxon_df.empty:
        print("No Wilcoxon results.")
    else:
        print(wilcoxon_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description='Exploratory paired analysis; overlapping splits are not independent datasets')
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results_dir = args.results_dir
    args.output.mkdir(parents=True, exist_ok=False)
    print(f"Using results directory: {results_dir}")

    df = load_detailed_results(results_dir)

    summary_df = build_mean_std_table(df)
    friedman_df = run_friedman_by_classifier(df)
    wilcoxon_df = run_pairwise_wilcoxon(df)

    save_dataframe(summary_df, args.output / "stat_summary_mean_std.csv")
    save_dataframe(friedman_df, args.output / "friedman_test_results.csv")
    save_dataframe(wilcoxon_df, args.output / "wilcoxon_pairwise_results.csv")

    print_pretty_summary(friedman_df, wilcoxon_df, summary_df)

    print("\nSaved:")
    print(args.output / "stat_summary_mean_std.csv")
    print(args.output / "friedman_test_results.csv")
    print(args.output / "wilcoxon_pairwise_results.csv")


if __name__ == "__main__":
    main()
