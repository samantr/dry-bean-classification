import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from ga_feature_selection import _build_toolbox, run_ga_feature_selection


@pytest.fixture
def sample():
    return make_classification(n_samples=90, n_features=6, n_informative=4, random_state=19)


def test_fitness_matches_fold_local_pipeline(sample):
    X, y = sample
    X[:, 0] *= 10000
    toolbox = _build_toolbox(X, y, 7, 6, 0.1, 3)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=7))
    expected = cross_val_score(model, X[:, :3], y, scoring='f1_macro',
                              cv=StratifiedKFold(3, shuffle=True, random_state=7)).mean()
    assert toolbox.evaluate([1, 1, 1, 0, 0, 0])[0] == pytest.approx(expected - 0.0005)
    assert toolbox.evaluate([0] * 6) == (0.0,)


def test_each_scaler_sees_only_inner_training_rows(sample, monkeypatch):
    X, y = sample
    seen = []
    original = StandardScaler.fit
    def recording_fit(self, X, *args, **kwargs):
        seen.append(len(X))
        return original(self, X, *args, **kwargs)
    monkeypatch.setattr(StandardScaler, 'fit', recording_fit)
    _build_toolbox(X, y, 7, 6, 0.1, 3).evaluate([1] * 6)
    assert seen == [60, 60, 60]


@pytest.mark.parametrize('variant', ['vanilla_ga', 'improved_ga'])
def test_reproducible_and_scores_separate(sample, variant):
    X, y = sample
    a = run_ga_feature_selection(X, y, population_size=6, generations=2, ga_variant=variant)
    b = run_ga_feature_selection(X, y, population_size=6, generations=2, ga_variant=variant)
    assert a['selected_features'] == b['selected_features']
    assert a['best_score'] == b['best_score']
    assert a['meta']['ga_best_cv_macro_f1'] == pytest.approx(
        a['meta']['ga_best_penalized_fitness'] + 0.001 * len(a['selected_features']) / 6)


def test_invalid_configuration(sample):
    X, y = sample
    with pytest.raises(ValueError):
        run_ga_feature_selection(X, y, population_size=0)
    with pytest.raises(ValueError):
        run_ga_feature_selection(X, y, ga_variant='unknown')
