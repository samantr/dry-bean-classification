from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import numpy as np
import pandas as pd

from baseline_models import train_and_evaluate_classifiers
from dimensionality_reduction import apply_nca
from ga_feature_selection import run_ga_feature_selection
from preprocessing import load_and_preprocess_data
from results_utils import (
    build_ablation_results,
    build_average_and_std_tables,
    build_feature_stability_summary,
    ensure_dir,
    save_rows_to_csv,
)


def flatten_classifier_results(
    seed,
    method,
    classifier_results,
    input_feature_count,
    output_feature_count,
    reduction_runtime_seconds=0.0,
    extra_meta=None,
):
    extra_meta = extra_meta or {}
    rows = []
    for classifier_name, metrics in classifier_results.items():
        rows.append({
            'seed': seed,
            'method': method,
            'classifier': classifier_name,
            'macro_f1': round(metrics['macro_f1'], 6),
            'accuracy': round(metrics['accuracy'], 6),
            'precision_macro': round(metrics['precision_macro'], 6),
            'recall_macro': round(metrics['recall_macro'], 6),
            'fit_time_seconds': round(metrics['fit_time_seconds'], 6),
            'predict_time_seconds': round(metrics['predict_time_seconds'], 6),
            'total_time_seconds': round(metrics['total_time_seconds'] + reduction_runtime_seconds, 6),
            'classifier_only_total_time_seconds': round(metrics['total_time_seconds'], 6),
            'reduction_runtime_seconds': round(reduction_runtime_seconds, 6),
            'input_feature_count': input_feature_count,
            'output_feature_count': output_feature_count,
            'feature_reduction_ratio': round((input_feature_count - output_feature_count) / input_feature_count, 6),
            **extra_meta,
        })
    return rows


def save_all_outputs(
    results_dir,
    long_results_rows,
    efficiency_rows,
    ga_feature_subsets_rows,
    ga_feature_stability_rows,
    ga_convergence_rows,
):
    results_dir = Path(results_dir)
    ensure_dir(results_dir)

    detailed_results_path = results_dir / 'detailed_results_long.csv'
    efficiency_results_path = results_dir / 'efficiency_results.csv'
    ga_feature_subsets_path = results_dir / 'ga_feature_subsets.csv'
    ga_feature_stability_path = results_dir / 'ga_feature_stability_matrix.csv'
    ga_convergence_path = results_dir / 'ga_convergence.csv'
    summary_average_path = results_dir / 'summary_average_results.csv'
    summary_std_path = results_dir / 'summary_std_results.csv'
    feature_stability_summary_path = results_dir / 'feature_stability_summary.csv'
    ablation_results_path = results_dir / 'ablation_results.csv'

    save_rows_to_csv(long_results_rows, detailed_results_path)
    save_rows_to_csv(efficiency_rows, efficiency_results_path)
    save_rows_to_csv(ga_feature_subsets_rows, ga_feature_subsets_path)
    save_rows_to_csv(ga_feature_stability_rows, ga_feature_stability_path)
    save_rows_to_csv(ga_convergence_rows, ga_convergence_path)

    if long_results_rows:
        df_long = pd.DataFrame(long_results_rows)
        avg_df, std_df = build_average_and_std_tables(df_long)
        avg_df.to_csv(summary_average_path, index=False)
        std_df.to_csv(summary_std_path, index=False)
        ablation_df = build_ablation_results(df_long)
        ablation_df.to_csv(ablation_results_path, index=False)
    else:
        avg_df = pd.DataFrame()
        std_df = pd.DataFrame()

    if ga_feature_stability_rows:
        df_stability = pd.DataFrame(ga_feature_stability_rows)
        feature_stability_summary_df = build_feature_stability_summary(df_stability)
        feature_stability_summary_df.to_csv(feature_stability_summary_path, index=False)

    return {
        'detailed_results_path': detailed_results_path,
        'efficiency_results_path': efficiency_results_path,
        'ga_feature_subsets_path': ga_feature_subsets_path,
        'ga_feature_stability_path': ga_feature_stability_path,
        'ga_convergence_path': ga_convergence_path,
        'summary_average_path': summary_average_path,
        'summary_std_path': summary_std_path,
        'feature_stability_summary_path': feature_stability_summary_path,
        'ablation_results_path': ablation_results_path,
        'avg_df': avg_df,
        'std_df': std_df,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Corrected repeated-holdout experiment; never overwrites a run.")
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 52, 62, 72, 82])
    parser.add_argument('--population', type=int, default=20)
    parser.add_argument('--generations', type=int, default=10)
    parser.add_argument('--skip-nca', action='store_true', help='Pilot only: omit expensive NCA')
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('Seeds must be unique')
    project_root = Path(__file__).resolve().parent.parent
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    RESULTS_DIR = args.output.resolve()

    # Exclusive creation protects archived results and partial runs alike.
    RESULTS_DIR.mkdir(parents=True, exist_ok=False)

    excel_files = list(DATA_DIR.glob("*.xlsx"))
    if not excel_files:
        raise FileNotFoundError(f"No .xlsx file found in {DATA_DIR}")

    file_path = str(excel_files[0])
    results_dir = str(RESULTS_DIR)

    print(f"Project root: {PROJECT_ROOT}", flush=True)
    print(f"Results directory: {RESULTS_DIR}", flush=True)

    print(f'Project root: {project_root}', flush=True)
    print(f'Results directory: {results_dir}', flush=True)

    random_seeds = args.seeds
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted((PROJECT_ROOT / 'src').glob('*.py'))}
    manifest = {
        'protocol': 'fold_local_scaling_v1', 'status': 'running',
        'seeds': random_seeds, 'population': args.population, 'generations': args.generations,
        'skip_nca': args.skip_nca, 'test_fraction': 0.2, 'inner_folds': 3,
        'dataset_sha256': hashlib.sha256(Path(file_path).read_bytes()).hexdigest(),
        'source_sha256': source_hashes,
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT, text=True).strip(),
        'git_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=PROJECT_ROOT, text=True).strip()),
        'python': platform.python_version(),
        'packages': {name: importlib.metadata.version(name) for name in
                     ['numpy', 'pandas', 'scipy', 'scikit-learn', 'deap', 'openpyxl']},
        'timing_note': 'Model fit/predict plus selector cost; excludes shared loading/scaling. Not end-to-end deployment latency.'
    }
    (RESULTS_DIR / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    ga_variants = ['vanilla_ga', 'improved_ga']

    long_results_rows = []
    efficiency_rows = []
    ga_feature_subsets_rows = []
    ga_feature_stability_rows = []
    ga_convergence_rows = []

    for seed in random_seeds:
        print('=' * 90, flush=True)
        print(f'RUNNING EXPERIMENT FOR RANDOM SEED = {seed}', flush=True)
        print('=' * 90, flush=True)

        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, label_encoder = load_and_preprocess_data(
            file_path=file_path,
            test_size=0.2,
            random_state=seed,
        )

        feature_names = list(X_train.columns)
        (RESULTS_DIR / f'split_{seed}.json').write_text(json.dumps({
            'train_indices': X_train.index.tolist(), 'test_indices': X_test.index.tolist(),
            'classes': label_encoder.classes_.tolist(), 'features': feature_names,
        }, indent=2))
        input_feature_count = len(feature_names)

        X_train_np = np.asarray(X_train)
        X_test_np = np.asarray(X_test)
        X_train_scaled_np = np.asarray(X_train_scaled)
        X_test_scaled_np = np.asarray(X_test_scaled)

        baseline_results = train_and_evaluate_classifiers(
            X_train_np,
            X_test_np,
            X_train_scaled_np,
            X_test_scaled_np,
            y_train,
            y_test,
            random_state=seed,
        )

        print('\nBaseline Macro-F1 Results:', flush=True)
        for model_name, metrics in baseline_results.items():
            print(f"{model_name}: {metrics['macro_f1']:.4f}", flush=True)

        long_results_rows.extend(
            flatten_classifier_results(
                seed=seed,
                method='baseline',
                classifier_results=baseline_results,
                input_feature_count=input_feature_count,
                output_feature_count=input_feature_count,
            )
        )

        for ga_variant in ga_variants:
            print(f'\nStarting {ga_variant} ...', flush=True)
            ga_output = run_ga_feature_selection(
                X_train=X_train_np,
                y_train=y_train,
                random_state=seed,
                population_size=args.population,
                generations=args.generations,
                cxpb=0.8,
                mutpb=0.1,
                ga_variant=ga_variant,
            )

            selected_features = ga_output['selected_features']
            ga_meta = ga_output['meta']
            ga_convergence_rows.extend(ga_output['convergence_rows'])

            print(f'\n{ga_variant} Feature Selection Results:', flush=True)
            print('Selected feature indices:', selected_features, flush=True)
            print('Number of selected features:', len(selected_features), flush=True)
            print(f"Best penalized GA fitness: {ga_output['best_score']:.4f}", flush=True)

            X_train_ga = X_train_np[:, selected_features]
            X_test_ga = X_test_np[:, selected_features]
            X_train_scaled_ga = X_train_scaled_np[:, selected_features]
            X_test_scaled_ga = X_test_scaled_np[:, selected_features]

            ga_results = train_and_evaluate_classifiers(
                X_train_ga,
                X_test_ga,
                X_train_scaled_ga,
                X_test_scaled_ga,
                y_train,
                y_test,
                random_state=seed,
            )

            print(f'\n{ga_variant} Macro-F1 Results:', flush=True)
            for model_name, metrics in ga_results.items():
                print(f"{model_name}: {metrics['macro_f1']:.4f}", flush=True)

            extra_meta = {
                'ga_variant': ga_variant,
                'ga_best_cv_macro_f1': round(ga_meta['ga_best_cv_macro_f1'], 6),
                'ga_best_penalized_fitness': round(ga_meta['ga_best_penalized_fitness'], 6),
                'fitness_classifier': ga_meta['fitness_classifier'],
                'fitness_metric': ga_meta['fitness_metric'],
                'population_size': ga_meta['population_size'],
                'generations': ga_meta['generations'],
                'cxpb': ga_meta['cxpb'],
                'mutpb': ga_meta['mutpb'],
                'mutation_indpb': ga_meta['mutation_indpb'],
                'tournament_size': ga_meta['tournament_size'],
            }
            long_results_rows.extend(
                flatten_classifier_results(
                    seed=seed,
                    method=ga_variant,
                    classifier_results=ga_results,
                    input_feature_count=input_feature_count,
                    output_feature_count=len(selected_features),
                    reduction_runtime_seconds=ga_meta['ga_runtime_seconds'],
                    extra_meta=extra_meta,
                )
            )

            efficiency_rows.append({
                'seed': seed,
                'method': ga_variant,
                **ga_meta,
            })

            selected_feature_names = [feature_names[idx] for idx in selected_features]
            ga_feature_subsets_rows.append({
                'seed': seed,
                'ga_variant': ga_variant,
                'selected_feature_indices': ','.join(map(str, selected_features)),
                'selected_feature_names': '|'.join(selected_feature_names),
                'selected_feature_count': len(selected_features),
            })

            selected_set = set(selected_features)
            for feature_idx, feature_name in enumerate(feature_names):
                ga_feature_stability_rows.append({
                    'seed': seed,
                    'ga_variant': ga_variant,
                    'feature_index': feature_idx,
                    'feature_name': feature_name,
                    'selected': 1 if feature_idx in selected_set else 0,
                })

            outputs = save_all_outputs(
                results_dir,
                long_results_rows,
                efficiency_rows,
                ga_feature_subsets_rows,
                ga_feature_stability_rows,
                ga_convergence_rows,
            )
            print(f'Saved partial results after {ga_variant} for seed {seed} to: {results_dir}', flush=True)

        if args.skip_nca:
            continue
        n_classes = len(np.unique(y_train))
        nca_components = n_classes - 1
        print(f'\nStarting NCA with n_components={nca_components} ...', flush=True)
        X_train_nca, X_test_nca, _, nca_meta = apply_nca(
            X_train_scaled_np,
            X_test_scaled_np,
            y_train,
            n_components=nca_components,
            random_state=seed,
        )

        nca_results = train_and_evaluate_classifiers(
            X_train_nca,
            X_test_nca,
            X_train_nca,
            X_test_nca,
            y_train,
            y_test,
            random_state=seed,
        )

        print('\nNCA Macro-F1 Results:', flush=True)
        for model_name, metrics in nca_results.items():
            print(f"{model_name}: {metrics['macro_f1']:.4f}", flush=True)

        long_results_rows.extend(
            flatten_classifier_results(
                seed=seed,
                method='nca',
                classifier_results=nca_results,
                input_feature_count=input_feature_count,
                output_feature_count=nca_components,
                reduction_runtime_seconds=nca_meta['total_time_seconds'],
                extra_meta={
                    'nca_components': nca_components,
                },
            )
        )

        efficiency_rows.append({
            'seed': seed,
            'method': 'nca',
            'input_feature_count': input_feature_count,
            'selected_feature_count': nca_components,
            'feature_reduction_ratio': round((input_feature_count - nca_components) / input_feature_count, 6),
            'ga_best_cv_macro_f1': None,
            'ga_runtime_seconds': None,
            'population_size': None,
            'generations': None,
            'cxpb': None,
            'mutpb': None,
            'mutation_indpb': None,
            'tournament_size': None,
            'fitness_classifier': None,
            'fitness_metric': None,
            'fit_time_seconds': round(nca_meta['fit_time_seconds'], 6),
            'transform_time_seconds': round(nca_meta['transform_time_seconds'], 6),
            'total_time_seconds': round(nca_meta['total_time_seconds'], 6),
            'n_components': nca_meta['n_components'],
        })

        outputs = save_all_outputs(
            results_dir,
            long_results_rows,
            efficiency_rows,
            ga_feature_subsets_rows,
            ga_feature_stability_rows,
            ga_convergence_rows,
        )
        print(f'Completed seed {seed}. Partial results safely saved to: {results_dir}', flush=True)

    manifest['status'] = 'complete'
    (RESULTS_DIR / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    avg_df = outputs['avg_df']

    print('\n' + '=' * 90, flush=True)
    print('FINAL AVERAGE RESULTS ACROSS SEEDS', flush=True)
    print('=' * 90, flush=True)
    for _, row in avg_df.iterrows():
        print(f"{row['classifier']} | {row['method']} | macro_f1={row['macro_f1']:.4f}", flush=True)

    print('\nSaved files:', flush=True)
    for path in [
        outputs['detailed_results_path'],
        outputs['efficiency_results_path'],
        outputs['ga_feature_subsets_path'],
        outputs['ga_feature_stability_path'],
        outputs['ga_convergence_path'],
        outputs['summary_average_path'],
        outputs['summary_std_path'],
        outputs['feature_stability_summary_path'],
        outputs['ablation_results_path'],
    ]:
        print(path, flush=True)
