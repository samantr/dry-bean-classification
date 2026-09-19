"""Training-only, unique-evaluation-budget search for this <=20-feature study.

This is a new budget-controlled protocol, not a reproduction of legacy eaSimple.
All randomness is local. Cache hits never consume an additional CV evaluation.
"""
import random

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


GA_SETTINGS = {
    'ga_standard': (0.1, 3),
    'ga_mutation_only': (0.2, 3),
    'ga_tournament_only': (0.1, 5),
    'ga_both': (0.2, 5),
}
SEARCH_METHODS = ('random_search', *GA_SETTINGS)


def mask_indices(mask, dimension):
    return [i for i in range(dimension) if mask & (1 << i)]


class SubsetEvaluator:
    def __init__(self, X, y, seed, penalty=0.001):
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        if self.X.ndim != 2 or len(self.X) != len(self.y):
            raise ValueError('X and y must be aligned')
        self.dimension = self.X.shape[1]
        if not 2 <= self.dimension <= 20:
            raise ValueError('This study-specific search supports 2..20 input features')
        if penalty < 0 or not np.isfinite(penalty):
            raise ValueError('penalty must be finite and nonnegative')
        if np.unique(self.y, return_counts=True)[1].min() < 3:
            raise ValueError('Each training class needs at least three rows')
        self.seed, self.penalty = seed, penalty
        self.folds = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(self.X, self.y))
        self.cache = {}
        self.history = []
        self.cache_hits = 0

    def evaluate(self, mask):
        if not 1 <= mask < (1 << self.dimension):
            raise ValueError('Expected a valid nonempty subset mask')
        if mask in self.cache:
            self.cache_hits += 1
            return self.cache[mask]
        indices = mask_indices(mask, self.dimension)
        estimator = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=self.seed))
        scores = cross_val_score(estimator, self.X[:, indices], self.y, cv=self.folds,
                                 scoring='f1_macro', n_jobs=1, error_score='raise')
        cv_mean = float(np.mean(scores))
        record = {'mask': int(mask), 'selected_count': len(indices),
                  'cv_macro_f1': cv_mean, 'fold_scores': scores.tolist(),
                  'fitness': cv_mean - self.penalty * len(indices) / self.dimension}
        self.cache[mask] = record
        self.history.append(dict(record, evaluation=len(self.cache)))
        return record


def budgeted_search(X, y, method, seed, budget=200, population=20, evaluator=None):
    if method not in SEARCH_METHODS:
        raise ValueError(f'Unknown search method: {method}')
    scorer = evaluator if evaluator is not None else SubsetEvaluator(X, y, seed)
    d = scorer.dimension
    if not 2 <= population <= budget <= (1 << d) - 1:
        raise ValueError('Require 2 <= population <= budget <= number of nonempty subsets')
    if scorer.cache:
        raise ValueError('Each method requires a fresh evaluator cache')
    rng = random.Random(seed)
    # Identical initial subset sequence across all methods for the same seed.
    unseen_order = list(range(1, 1 << d))
    rng.shuffle(unseen_order)
    cursor = 0

    def next_unseen():
        nonlocal cursor
        while unseen_order[cursor] in scorer.cache:
            cursor += 1
        result = unseen_order[cursor]
        cursor += 1
        return result

    def key(mask):
        r = scorer.cache[mask]
        return (r['fitness'], -r['selected_count'], -mask)

    pool = []
    for _ in range(population):
        mask = next_unseen()
        scorer.evaluate(mask)
        pool.append(mask)
    generations, rescues, empty_repairs = 0, 0, 0
    if method == 'random_search':
        while len(scorer.cache) < budget:
            scorer.evaluate(next_unseen())
    else:
        mutation, tournament = GA_SETTINGS[method]
        stagnant = 0
        while len(scorer.cache) < budget:
            before = len(scorer.cache)
            children = [max(rng.choices(pool, k=tournament), key=key) for _ in range(population)]
            for i in range(1, population, 2):
                if rng.random() < 0.8:
                    # Standard two-point crossover: cut positions 1..d.
                    left, right = sorted(rng.sample(range(1, d + 1), 2))
                    segment = ((1 << right) - 1) ^ ((1 << left) - 1)
                    a, b = children[i - 1], children[i]
                    children[i - 1] = (a & ~segment) | (b & segment)
                    children[i] = (b & ~segment) | (a & segment)
            for i, mask in enumerate(children):
                if rng.random() < 0.1:
                    for bit in range(d):
                        if rng.random() < mutation:
                            mask ^= 1 << bit
                if mask == 0:
                    mask = 1 << rng.randrange(d)
                    empty_repairs += 1
                children[i] = mask
                scorer.evaluate(mask)
                if len(scorer.cache) == budget:
                    break
            generations += 1
            if len(scorer.cache) == budget:
                break
            pool = children
            stagnant = stagnant + 1 if before == len(scorer.cache) else 0
            if stagnant >= 10:
                # Explicitly logged safety rule, common to every GA variant.
                # It prevents unlimited duplicate-only generations.
                immigrant = next_unseen()
                scorer.evaluate(immigrant)
                pool[0] = immigrant
                rescues += 1
                stagnant = 0

    best = max(scorer.cache, key=key)
    return {
        'selected_indices': mask_indices(best, d),
        'best': scorer.cache[best], 'history': scorer.history,
        'unique_evaluations': len(scorer.cache), 'cv_model_fits': 3 * len(scorer.cache),
        'cache_hits': scorer.cache_hits, 'generations': generations,
        'stagnation_immigrants': rescues, 'empty_subset_repairs': empty_repairs,
        'mutation_indpb': GA_SETTINGS[method][0] if method in GA_SETTINGS else None,
        'tournament_size': GA_SETTINGS[method][1] if method in GA_SETTINGS else None,
    }
