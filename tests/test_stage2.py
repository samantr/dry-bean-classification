import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from stage2 import atomic_json, digest, execute_unit, run, worker


def test_timeout_is_checkpointed_and_not_retried(tmp_path):
    command = [sys.executable, '-c', 'import time; time.sleep(10)']
    status = execute_unit(command, tmp_path, 42, 'nca6', timeout=0.05)
    assert status['state'] == 'timeout'
    assert execute_unit(['invalid-command'], tmp_path, 42, 'nca6', timeout=0.05) == status
    assert not (tmp_path / 'result.json').exists()


def test_complete_checkpoint_verified_on_resume(tmp_path):
    atomic_json(tmp_path / 'result.json', {'seed': 42, 'method': 'baseline'})
    status = {'seed': 42, 'method': 'baseline', 'state': 'complete',
              'result_sha256': digest(tmp_path / 'result.json')}
    atomic_json(tmp_path / 'status.json', status)
    assert execute_unit(['invalid-command'], tmp_path, 42, 'baseline', 1) == status
    atomic_json(tmp_path / 'result.json', {'seed': 43, 'method': 'baseline'})
    with pytest.raises(ValueError, match='verification failed'):
        execute_unit(['invalid-command'], tmp_path, 42, 'baseline', 1)


def test_retry_cannot_accept_stale_result(tmp_path):
    atomic_json(tmp_path / 'status.json', {'state': 'timeout', 'attempt': 1})
    atomic_json(tmp_path / 'result.json', {'seed': 42, 'method': 'baseline'})
    status = execute_unit([sys.executable, '-c', 'pass'], tmp_path, 42, 'baseline', 5, retry=True)
    assert status['state'] == 'error'
    assert (tmp_path / 'result-before-attempt-2.json').exists()
    assert (tmp_path / 'status-attempt-1.json').exists()


@pytest.fixture
def config(tmp_path):
    X, y = make_classification(n_samples=90, n_features=10, n_informative=5, random_state=5)
    frame = pd.DataFrame(X, columns=[f'f{i}' for i in range(10)])
    frame['Class'] = y.astype(str)
    path = tmp_path / 'data.xlsx'
    frame.to_excel(path, index=False)
    return {'dataset': str(path), 'seeds': [42], 'methods': ['baseline'], 'budget': 8,
            'population': 4, 'unit_timeout': 20, 'nca_timeout': 20}


def test_runner_end_to_end_resume_and_provenance(config, tmp_path):
    destination = tmp_path / 'run'
    assert run(destination, config) == 'complete'
    result = destination / 'units/42-baseline/result.json'
    stamp = result.stat().st_mtime_ns
    assert run(destination, config, resume=True) == 'complete'
    assert result.stat().st_mtime_ns == stamp
    with pytest.raises(ValueError, match='Resume rejected'):
        run(destination, dict(config, budget=9), resume=True)
    with pytest.raises(FileExistsError):
        run(destination, config)
    data = json.loads(result.read_text())
    assert not set(data['train_indices']) & set(data['test_indices'])
    assert len(data['train_indices']) + len(data['test_indices']) == 90
    for metrics in data['classifiers'].values():
        assert len(metrics['predictions']) == len(data['y_true'])
    summary = pd.read_csv(destination / 'summary.csv')
    assert summary.completed_seeds.tolist() == [1, 1, 1]


def test_all_timeouts_have_missing_scores_and_resume_skips(config, tmp_path):
    limited = dict(config, methods=['baseline', 'nca6'], unit_timeout=0.001, nca_timeout=0.001)
    destination = tmp_path / 'timeout-run'
    assert run(destination, limited) == 'incomplete'
    summary = pd.read_csv(destination / 'summary.csv')
    assert summary.completed_seeds.sum() == 0
    assert summary.mean_macro_f1.isna().all()
    assert run(destination, limited, resume=True) == 'incomplete'
    statuses = json.loads((destination / 'unit-statuses.json').read_text())
    assert all(s['state'] == 'timeout' and s['attempt'] == 1 for s in statuses)


def test_live_lock_blocks_resume(config, tmp_path):
    destination = tmp_path / 'locked-run'
    assert run(destination, config) == 'complete'
    (destination / 'runner.lock').write_text('another runner')
    with pytest.raises(FileExistsError):
        run(destination, config, resume=True)
    assert (destination / 'runner.lock').read_text() == 'another runner'


@pytest.mark.parametrize('method', ['mi8', 'rfe8', 'pca6', 'nca6', 'random_search'])
def test_worker_baselines(config, tmp_path, method):
    worker(config, 42, method, tmp_path)
    result = json.loads((tmp_path / 'result.json').read_text())
    assert set(result['classifiers']) == {'Logistic Regression', 'Random Forest', 'SVM'}
    if method in ('pca6', 'nca6'):
        assert result['required_original_inputs'] == 10
        assert result['output_dimensions'] == 6
    if method in ('mi8', 'rfe8'):
        assert result['required_original_inputs'] == result['output_dimensions'] == 8
    if method == 'random_search':
        assert result['search']['unique_evaluations'] == 8
