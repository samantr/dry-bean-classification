"""Bounded, sequential experiment units with strict provenance-aware resume."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('baseline', 'random_search', 'ga_standard', 'ga_mutation_only',
           'ga_tournament_only', 'ga_both', 'mi8', 'rfe8', 'pca6', 'nca6')
THREAD_ENV = {name: '1' for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                   'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def identity(config):
    return {'protocol': 'stage2_v1', 'config': config,
            'dataset_sha256': digest(config['dataset']),
            'source_sha256': {p.name: digest(p) for p in sorted((ROOT / 'src').glob('*.py'))},
            'python': platform.python_version(), 'platform': platform.platform(),
            'packages': {p: importlib.metadata.version(p) for p in
                         ('numpy', 'pandas', 'scipy', 'scikit-learn', 'openpyxl')},
            'thread_environment': THREAD_ENV}


def worker(config, seed, method, destination):
    # Heavy imports occur only in the bounded subprocess.
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import RFE, mutual_info_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 classification_report, confusion_matrix, f1_score)
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import NeighborhoodComponentsAnalysis
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from baseline_models import CLASSIFIERS
    from budgeted_search import SEARCH_METHODS, budgeted_search

    frame = pd.read_excel(config['dataset'])
    features = frame.drop(columns='Class')
    X = features.to_numpy(dtype=float)
    encoder = LabelEncoder()
    y = encoder.fit_transform(frame['Class'])
    train, test = train_test_split(np.arange(len(y)), test_size=0.2,
                                   stratify=y, random_state=seed)
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
    selection = None
    selected = list(range(X.shape[1]))
    start = time.perf_counter()
    if method in SEARCH_METHODS:
        selection = budgeted_search(X_train, y_train, method, seed,
                                    budget=config['budget'], population=config['population'])
        selected = selection['selected_indices']
        train_rep, test_rep = X_train[:, selected], X_test[:, selected]
    elif method == 'mi8':
        scores = mutual_info_classif(X_train, y_train, random_state=seed)
        selected = sorted(np.argsort(-scores, kind='stable')[:min(8, X.shape[1])].tolist())
        train_rep, test_rep = X_train[:, selected], X_test[:, selected]
    elif method == 'rfe8':
        estimator = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=seed))
        selector = RFE(estimator, n_features_to_select=min(8, X.shape[1]), step=1,
                       importance_getter='named_steps.logisticregression.coef_')
        selector.fit(X_train, y_train)
        selected = np.flatnonzero(selector.support_).tolist()
        train_rep, test_rep = X_train[:, selected], X_test[:, selected]
    elif method in ('pca6', 'nca6'):
        components = min(6, X.shape[1])
        transform = (PCA(n_components=components, random_state=seed) if method == 'pca6'
                     else NeighborhoodComponentsAnalysis(n_components=components,
                                                          max_iter=200, random_state=seed))
        transformer = make_pipeline(StandardScaler(), transform)
        train_rep = transformer.fit_transform(X_train, y_train)
        test_rep = transformer.transform(X_test)
    elif method == 'baseline':
        train_rep, test_rep = X_train, X_test
    else:
        raise ValueError(f'Unknown method: {method}')
    representation_seconds = time.perf_counter() - start
    result = {'seed': seed, 'method': method, 'train_indices': train.tolist(),
              'test_indices': test.tolist(), 'classes': encoder.classes_.tolist(),
              'selected_original_indices': selected,
              'selected_original_names': features.columns[selected].tolist(),
              'required_original_inputs': len(selected), 'output_dimensions': train_rep.shape[1],
              'representation_seconds': representation_seconds, 'search': selection,
              'y_true': y_test.tolist(), 'classifiers': {}}
    for name, factory in CLASSIFIERS.items():
        model = factory(seed)
        if name != 'Random Forest':
            model = make_pipeline(StandardScaler(), model)
        start = time.perf_counter()
        model.fit(train_rep, y_train)
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        prediction = model.predict(test_rep)
        predict_seconds = time.perf_counter() - start
        result['classifiers'][name] = {
            'macro_f1': float(f1_score(y_test, prediction, average='macro')),
            'accuracy': float(accuracy_score(y_test, prediction)),
            'balanced_accuracy': float(balanced_accuracy_score(y_test, prediction)),
            'fit_seconds': fit_seconds, 'predict_seconds': predict_seconds,
            'predictions': prediction.tolist(),
            'per_class': classification_report(y_test, prediction, output_dict=True,
                                               target_names=[str(c) for c in encoder.classes_], zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, prediction).tolist()}
    atomic_json(Path(destination) / 'result.json', result)


def verified_result(unit, status):
    path = unit / 'result.json'
    if not path.exists() or digest(path) != status.get('result_sha256'):
        raise ValueError(f'Checkpoint verification failed: {unit}')
    result = json.loads(path.read_text())
    if (result['seed'], result['method']) != (status['seed'], status['method']):
        raise ValueError(f'Checkpoint identity mismatch: {unit}')
    return result


def execute_unit(command, unit, seed, method, timeout, retry=False):
    unit.mkdir(parents=True, exist_ok=True)
    status_path = unit / 'status.json'
    previous = json.loads(status_path.read_text()) if status_path.exists() else None
    if previous and previous['state'] == 'complete':
        verified_result(unit, previous)
        return previous
    if previous and not retry and previous['state'] in ('timeout', 'error'):
        return previous
    attempt = previous.get('attempt', 0) + 1 if previous else 1
    if previous:
        atomic_json(unit / f'status-attempt-{attempt - 1}.json', previous)
    if (unit / 'result.json').exists():
        (unit / 'result.json').rename(unit / f'result-before-attempt-{attempt}.json')
    status = {'seed': seed, 'method': method, 'state': 'running', 'attempt': attempt,
              'timeout_seconds': timeout}
    atomic_json(status_path, status)
    started = time.perf_counter()
    # A file-backed log remains available even when the worker is killed.
    try:
        with (unit / f'attempt-{attempt}.log').open('w', encoding='utf-8') as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                       env={**os.environ, **THREAD_ENV}, timeout=timeout)
        if completed.returncode:
            status.update(state='error', returncode=completed.returncode)
        else:
            path = unit / 'result.json'
            if not path.exists():
                raise ValueError('Worker returned successfully without a result')
            status.update(state='complete', result_sha256=digest(path))
            verified_result(unit, status)
    except subprocess.TimeoutExpired:
        status['state'] = 'timeout'
    except KeyboardInterrupt:
        status.update(state='interrupted', elapsed_seconds=time.perf_counter() - started)
        atomic_json(status_path, status)
        raise
    except Exception as error:
        status.update(state='error', error=str(error))
    status['elapsed_seconds'] = time.perf_counter() - started
    atomic_json(status_path, status)
    return status


def summarize(output, config):
    statuses, rows = [], []
    for seed in config['seeds']:
        for method in config['methods']:
            unit = output / 'units' / f'{seed}-{method}'
            status = json.loads((unit / 'status.json').read_text()) if (unit / 'status.json').exists() else {
                'seed': seed, 'method': method, 'state': 'pending'}
            statuses.append(status)
            if status['state'] == 'complete':
                result = verified_result(unit, status)
                for classifier, metrics in result['classifiers'].items():
                    rows.append({'seed': seed, 'method': method, 'classifier': classifier,
                                 'macro_f1': metrics['macro_f1'], 'accuracy': metrics['accuracy'],
                                 'balanced_accuracy': metrics['balanced_accuracy'],
                                 'required_original_inputs': result['required_original_inputs'],
                                 'output_dimensions': result['output_dimensions'],
                                 'representation_seconds': result['representation_seconds'],
                                 'fit_seconds': metrics['fit_seconds'],
                                 'predict_seconds': metrics['predict_seconds']})
    atomic_json(output / 'unit-statuses.json', statuses)
    fields = list(rows[0]) if rows else ['seed', 'method', 'classifier', 'macro_f1']
    with (output / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summaries = []
    for method in config['methods']:
        for classifier in ('Logistic Regression', 'Random Forest', 'SVM'):
            paired = [r for r in rows if r['method'] == method and r['classifier'] == classifier]
            scores = [r['macro_f1'] for r in paired]
            summaries.append({'method': method, 'classifier': classifier,
                              'completed_seeds': len(scores), 'requested_seeds': len(config['seeds']),
                              'mean_macro_f1': statistics.mean(scores) if scores else None,
                              'std_macro_f1': statistics.stdev(scores) if len(scores) > 1 else None,
                              'mean_original_inputs': statistics.mean(r['required_original_inputs'] for r in paired) if paired else None,
                              'mean_output_dimensions': statistics.mean(r['output_dimensions'] for r in paired) if paired else None})
    with (output / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    return all(s['state'] == 'complete' for s in statuses)


def run(output, config, resume=False, retry=False):
    expected = identity(config)
    if resume:
        previous = json.loads((output / 'manifest.json').read_text())
        if previous['identity'] != expected:
            raise ValueError('Resume rejected: configuration, data, source or environment changed')
    else:
        output.mkdir(parents=True, exist_ok=False)
    lock = output / 'runner.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    manifest = {'identity': expected, 'state': 'running'}
    try:
        atomic_json(output / 'manifest.json', manifest)
        atomic_json(output / 'config.json', config)
        for seed in config['seeds']:
            for method in config['methods']:
                if identity(config) != expected:
                    raise ValueError('Inputs or source changed during the run')
                unit = output / 'units' / f'{seed}-{method}'
                command = [sys.executable, str(Path(__file__).resolve()), '--worker',
                           str(output / 'config.json'), str(seed), method, str(unit)]
                timeout = config['nca_timeout'] if method == 'nca6' else config['unit_timeout']
                status = execute_unit(command, unit, seed, method, timeout, retry)
                print(f'{seed} {method}: {status["state"]}', flush=True)
                summarize(output, config)
        manifest['state'] = 'complete' if summarize(output, config) else 'incomplete'
    except KeyboardInterrupt:
        manifest['state'] = 'interrupted'
        raise
    except Exception:
        manifest['state'] = 'error'
        raise
    finally:
        atomic_json(output / 'manifest.json', manifest)
        lock.unlink()
    return manifest['state']


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        config_path, seed, method, destination = sys.argv[2:]
        worker(json.loads(Path(config_path).read_text()), int(seed), method, destination)
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'data/Dry_Bean_Dataset.xlsx')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 52])
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--budget', type=int, default=40)
    parser.add_argument('--population', type=int, default=10)
    parser.add_argument('--unit-timeout', type=float, default=90)
    parser.add_argument('--nca-timeout', type=float, default=30)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.methods)) != len(args.methods):
        parser.error('Seeds and methods must be unique')
    if (not 2 <= args.population <= args.budget or
            any(not math.isfinite(t) or t <= 0 for t in (args.unit_timeout, args.nca_timeout)) or
            any(s < 0 or s >= 2**32 for s in args.seeds)):
        parser.error('Invalid budget, population or timeout')
    if args.retry_failed and not args.resume:
        parser.error('--retry-failed requires --resume')
    config = {'dataset': str(args.dataset.resolve()), 'seeds': args.seeds, 'methods': args.methods,
              'budget': args.budget, 'population': args.population,
              'unit_timeout': args.unit_timeout, 'nca_timeout': args.nca_timeout}
    state = run(args.output.resolve(), config, args.resume, args.retry_failed)
    print(f'Run state: {state}', flush=True)
    return 0 if state == 'complete' else 2


if __name__ == '__main__':
    sys.exit(main())
