import importlib.util
from pathlib import Path

import pytest
from stage2 import atomic_json, digest

spec = importlib.util.spec_from_file_location('analysis', Path(__file__).resolve().parents[1] / 'scripts/analyze_stage3.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def fixture_run(root):
    atomic_json(root / 'manifest.json', {'identity': {'config': {
        'seeds': [1001, 1002], 'methods': ['random_search', 'ga_standard'], 'budget': 2}}})
    for seed, method, score in [(1001, 'random_search', .8), (1001, 'ga_standard', .9),
                                (1002, 'random_search', .7)]:
        unit = root / 'units' / f'{seed}-{method}'
        unit.mkdir(parents=True)
        metrics = dict(macro_f1=score, accuracy=score, balanced_accuracy=score,
                       fit_seconds=1, predict_seconds=.1)
        atomic_json(unit / 'result.json', dict(seed=seed, method=method,
                    train_indices=[0, 1], test_indices=[2], selected_original_names=['a'],
                    required_original_inputs=1, representation_seconds=2,
                    search=dict(unique_evaluations=2, cv_model_fits=6,
                                history=[{'mask': 1}, {'mask': 2}]),
                    classifiers={k: metrics for k in ('Logistic Regression', 'Random Forest', 'SVM')}))
        atomic_json(unit / 'status.json', dict(seed=seed, method=method, state='complete',
                    result_sha256=digest(unit / 'result.json')))


def test_pairing_does_not_impute_missing_runs(tmp_path):
    fixture_run(tmp_path)
    report = analysis.analyze(tmp_path)
    paired = report['paired_comparisons'][0]
    assert paired['included_seeds'] == [1001]
    assert paired['macro_f1_difference']['mean'] == pytest.approx(.1)
    assert paired['macro_f1_difference']['sd'] is None
    assert (paired['wins'], paired['ties'], paired['losses']) == (1, 0, 0)
    assert len(report['missing_units']) == 1
    assert report['subset_stability'][0]['pairwise_jaccard']['mean'] == 1


def test_analysis_rejects_tampered_checkpoint(tmp_path):
    fixture_run(tmp_path)
    path = tmp_path / 'units/1001-ga_standard/result.json'
    path.write_text('{}')
    with pytest.raises(ValueError, match='verification failed'):
        analysis.analyze(tmp_path)
