import time
from typing import Dict, Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import SVC


CLASSIFIERS = {
    "Logistic Regression": lambda random_state: LogisticRegression(max_iter=5000, random_state=random_state),
    "Random Forest": lambda random_state: RandomForestClassifier(n_estimators=200, random_state=random_state),
    "SVM": lambda random_state: SVC(kernel="rbf", random_state=random_state),
}

SCALED_CLASSIFIERS = {"Logistic Regression", "SVM"}


def _compute_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }



def train_and_evaluate_classifiers(
    X_train,
    X_test,
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
    random_state=42,
):
    """
    Returns a nested dictionary with metrics and timing per classifier.
    """
    results: Dict[str, Dict[str, Any]] = {}

    for classifier_name, classifier_factory in CLASSIFIERS.items():
        model = classifier_factory(random_state)

        if classifier_name in SCALED_CLASSIFIERS:
            train_X, test_X = X_train_scaled, X_test_scaled
        else:
            train_X, test_X = X_train, X_test

        fit_start = time.perf_counter()
        model.fit(train_X, y_train)
        fit_time = time.perf_counter() - fit_start

        predict_start = time.perf_counter()
        y_pred = model.predict(test_X)
        predict_time = time.perf_counter() - predict_start

        metrics = _compute_metrics(y_test, y_pred)
        results[classifier_name] = {
            **metrics,
            "fit_time_seconds": fit_time,
            "predict_time_seconds": predict_time,
            "total_time_seconds": fit_time + predict_time,
        }

    return results
