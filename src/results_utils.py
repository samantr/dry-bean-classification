import pandas as pd
import csv
import os
from typing import List

def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)

def save_rows_to_csv(rows: List[dict], output_path: str):
    if not rows:
        return

    ensure_dir(os.path.dirname(output_path))

    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def build_average_and_std_tables(df_long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_cols = [
        'macro_f1',
        'accuracy',
        'precision_macro',
        'recall_macro',
        'fit_time_seconds',
        'predict_time_seconds',
        'total_time_seconds',
    ]

    avg_df = (
        df_long.groupby(['method', 'classifier'])[metric_cols]
        .mean()
        .reset_index()
        .sort_values(['classifier', 'method'])
    )

    std_df = (
        df_long.groupby(['method', 'classifier'])[metric_cols]
        .std(ddof=1)
        .reset_index()
        .sort_values(['classifier', 'method'])
    )

    return avg_df, std_df


def build_feature_stability_summary(df_stability: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df_stability.groupby(['ga_variant', 'feature_index', 'feature_name'])['selected']
        .agg(['sum', 'count', 'mean'])
        .reset_index()
    )
    summary = summary.rename(columns={
        'sum': 'selected_times',
        'count': 'num_runs',
        'mean': 'selection_frequency',
    })
    return summary.sort_values(['ga_variant', 'selection_frequency', 'feature_index'], ascending=[True, False, True])


def build_ablation_results(df_long: pd.DataFrame) -> pd.DataFrame:
    ablation_methods = ['vanilla_ga', 'improved_ga']
    ablation_df = df_long[df_long['method'].isin(ablation_methods)].copy()
    return (
        ablation_df.groupby(['method', 'classifier'])[['macro_f1', 'accuracy', 'total_time_seconds']]
        .mean()
        .reset_index()
        .sort_values(['classifier', 'method'])
    )
