import random
import time
from typing import Dict, List, Tuple

import numpy as np
from deap import algorithms, base, creator, tools
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler



def _ensure_deap_creators():
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)



def _build_toolbox(
    X_train,
    y_train,
    random_state,
    n_features,
    mutation_indpb,
    tournament_size,
):
    _ensure_deap_creators()

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    scorer = make_scorer(f1_score, average="macro")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    def evaluate(individual):
        selected_count = sum(individual)
        if selected_count == 0:
            return (0.0,)

        selected_indices = [i for i, bit in enumerate(individual) if bit == 1]
        X_selected = X_train[:, selected_indices]
        # Fit preprocessing independently inside every inner training fold.
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=random_state))
        scores = cross_val_score(model, X_selected, y_train, cv=cv, scoring=scorer, n_jobs=None, error_score="raise")

        score = float(scores.mean())

        # Small penalty encourages smaller subsets without dominating macro-F1.
        penalty = 0.001 * (selected_count / n_features)
        return (score - penalty,)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=mutation_indpb)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)
    return toolbox



def _extract_logbook_rows(logbook, seed: int, ga_variant: str) -> List[Dict]:
    rows = []
    for row in logbook:
        rows.append({
            "seed": seed,
            "ga_variant": ga_variant,
            "generation": int(row["gen"]),
            "nevals": int(row.get("nevals", 0)),
            "avg_fitness": float(row.get("avg", 0.0)),
            "max_fitness": float(row.get("max", 0.0)),
        })
    return rows



def run_ga_feature_selection(
    X_train,
    y_train,
    random_state=42,
    population_size=20,
    generations=10,
    cxpb=0.8,
    mutpb=0.1,
    ga_variant="improved_ga",
):
    """
    X_train must contain raw, unscaled OUTER-training features only.
    Supported ga_variant values (legacy labels retained for traceability):
    - vanilla_ga
    - improved_ga
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    if X_train.ndim != 2 or X_train.shape[0] != len(y_train):
        raise ValueError("Expected a two-dimensional X_train aligned with y_train")
    n_features = X_train.shape[1]
    if n_features < 2 or population_size < 2 or generations < 0:
        raise ValueError("Need >=2 features, >=2 individuals, and >=0 generations")
    if not (0 <= cxpb <= 1 and 0 <= mutpb <= 1):
        raise ValueError("Crossover and mutation probabilities must be in [0, 1]")

    random.seed(random_state)
    np.random.seed(random_state)

    if ga_variant == "vanilla_ga":
        mutation_indpb = 0.1
        tournament_size = 3
    elif ga_variant == "improved_ga":
        mutation_indpb = 0.2
        tournament_size = 5
    else:
        raise ValueError(f"Unsupported ga_variant: {ga_variant}")

    toolbox = _build_toolbox(
        X_train=X_train,
        y_train=y_train,
        random_state=random_state,
        n_features=n_features,
        mutation_indpb=mutation_indpb,
        tournament_size=tournament_size,
    )

    population = toolbox.population(n=population_size)
    # At least one valid chromosome is necessary even for tiny pilot searches.
    if not any(sum(individual) for individual in population):
        population[0][0] = 1
    hall_of_fame = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    start = time.perf_counter()
    population, logbook = algorithms.eaSimple(
        population,
        toolbox,
        cxpb=cxpb,
        mutpb=mutpb,
        ngen=generations,
        stats=stats,
        halloffame=hall_of_fame,
        verbose=False,
    )
    ga_runtime_seconds = time.perf_counter() - start

    best_individual = hall_of_fame[0]
    selected_features = [i for i, bit in enumerate(best_individual) if bit == 1]
    best_score = float(best_individual.fitness.values[0])

    ga_meta = {
        "seed": random_state,
        "ga_variant": ga_variant,
        "fitness_classifier": "Logistic Regression",
        "fitness_metric": "macro_f1_cv_3fold_with_subset_penalty",
        "population_size": population_size,
        "generations": generations,
        "cxpb": cxpb,
        "mutpb": mutpb,
        "mutation_indpb": mutation_indpb,
        "tournament_size": tournament_size,
        "input_feature_count": n_features,
        "selected_feature_count": len(selected_features),
        "feature_reduction_ratio": (n_features - len(selected_features)) / n_features,
        "ga_best_cv_macro_f1": best_score + 0.001 * len(selected_features) / n_features,
        "ga_best_penalized_fitness": best_score,
        "ga_runtime_seconds": ga_runtime_seconds,
    }

    convergence_rows = _extract_logbook_rows(logbook, seed=random_state, ga_variant=ga_variant)

    return {
        "selected_features": selected_features,
        "best_score": best_score,
        "best_individual": list(best_individual),
        "logbook": logbook,
        "meta": ga_meta,
        "convergence_rows": convergence_rows,
    }
