import numpy as np

from algorithms.base_knn import KNNClassifier as BaseKNN
from algorithms.kd_knn import KNNClassifier as KDKNN


ITERS = 100
POINTS_COUNT = 1000
DIMENSIONALITY = 3
K = 5
METRIC = ("euclidean", 2)
BOUNDS = (-10, 10)
NOISE = 0.1
SEED = 24422442


def _generate_points(points_count, bounds, rng):
    points = np.empty((points_count, len(bounds)))
    for coordinate, (lower, upper) in enumerate(bounds):
        points[:, coordinate] = rng.uniform(lower, upper, points_count)
    return points


def _add_noise(points, noise, rng):
    noisy = points.copy()
    for coordinate in range(points.shape[1]):
        noisy[:, coordinate] += rng.normal(0, noise, len(points))
    return noisy


def _get_metrics(points_count=POINTS_COUNT, dimensionality=DIMENSIONALITY,
                 k=K, metric=METRIC):
    distance_metric, p = metric
    metrics = {
        "base_time": [],
        "kd_time": [],
        "base_calculations": [],
        "kd_calculations": [],
    }

    for iter in range(ITERS):
        rng = np.random.default_rng(SEED + iter)
        bounds = (BOUNDS,) * dimensionality

        X_train = _generate_points(points_count, bounds, rng)
        X_test = _generate_points(1, bounds, rng)
        X_test = _add_noise(X_test, NOISE, rng)
        # чёт только сейчас понял, что оно не нужно в данном эксперименте...
        y_train = np.zeros(points_count, dtype=int)

        for name, classifier in (("base", BaseKNN), ("kd", KDKNN)):
            classi = classifier(k=k, distance_metric=distance_metric, p=p)
            classi.fit(X_train, y_train)

            classi.predict(X_test)

            time, calculations = classi.metrics()
            metrics[f"{name}_time"].append(time)
            metrics[f"{name}_calculations"].append(calculations)

    return {
        name: float(np.mean(values))
        for name, values in metrics.items()
    }
