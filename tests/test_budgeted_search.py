import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from budgeted_search import SubsetEvaluator, budgeted_search, SEARCH_METHODS


class ToyEvaluator:
    dimension = 4
    def __init__(self):
        self.cache, self.history, self.cache_hits = {}, [], 0
    def evaluate(self, mask):
        if mask in self.cache:
            self.cache_hits += 1
            return self.cache[mask]
        count = mask.bit_count()
        value = {'mask': mask, 'selected_count': count, 'fitness': -abs(mask - 7)}
        self.cache[mask] = value
        self.history.append(value)
        return value


@pytest.mark.parametrize('method', SEARCH_METHODS)
def test_exact_unique_budget_and_known_optimum(method):
    result = budgeted_search(None, None, method, 5, budget=15, population=4, evaluator=ToyEvaluator())
    assert result['unique_evaluations'] == 15
    assert result['cv_model_fits'] == 45
    assert len({r['mask'] for r in result['history']}) == 15
    assert result['best']['mask'] == 7


def test_shared_initial_population_and_deterministic_runs():
    results = [budgeted_search(None, None, m, 42, budget=10, population=4, evaluator=ToyEvaluator())
               for m in SEARCH_METHODS]
    assert all(r['history'][:4] == results[0]['history'][:4] for r in results)
    assert results[-1] == budgeted_search(None, None, 'ga_both', 42, budget=10,
                                         population=4, evaluator=ToyEvaluator())


def test_no_extra_fit_on_cache_hit_and_fold_local_scaling(monkeypatch):
    X, y = make_classification(n_samples=60, n_features=4, random_state=3)
    observed = []
    fit = StandardScaler.fit
    def record(self, values, *args, **kwargs):
        observed.append(len(values))
        return fit(self, values, *args, **kwargs)
    monkeypatch.setattr(StandardScaler, 'fit', record)
    evaluator = SubsetEvaluator(X, y, 4)
    first = evaluator.evaluate(7)
    assert evaluator.evaluate(7) == first
    assert observed == [40, 40, 40]
    assert evaluator.cache_hits == 1
    assert first['fitness'] == pytest.approx(first['cv_macro_f1'] - 0.001 * 3 / 4)
    assert all(not set(a) & set(b) for a, b in evaluator.folds)


def test_invalid_budget():
    with pytest.raises(ValueError):
        budgeted_search(None, None, 'random_search', 1, budget=16, population=4,
                        evaluator=ToyEvaluator())
