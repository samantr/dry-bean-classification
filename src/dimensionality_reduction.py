import time

from sklearn.neighbors import NeighborhoodComponentsAnalysis



def apply_nca(X_train_scaled, X_test_scaled, y_train, n_components=6, random_state=42):
    fit_start = time.perf_counter()
    nca = NeighborhoodComponentsAnalysis(
        n_components=n_components,
        random_state=random_state,
        max_iter=200,
    )

    X_train_nca = nca.fit_transform(X_train_scaled, y_train)
    fit_time_seconds = time.perf_counter() - fit_start

    transform_start = time.perf_counter()
    X_test_nca = nca.transform(X_test_scaled)
    transform_time_seconds = time.perf_counter() - transform_start

    meta = {
        "method": "NCA",
        "fit_time_seconds": fit_time_seconds,
        "transform_time_seconds": transform_time_seconds,
        "total_time_seconds": fit_time_seconds + transform_time_seconds,
        "n_components": n_components,
    }

    return X_train_nca, X_test_nca, nca, meta
