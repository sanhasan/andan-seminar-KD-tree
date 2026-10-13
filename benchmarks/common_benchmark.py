import numpy as np

from algorithms.base_knn import KNNClassifier as BaseKNN
from algorithms.kd_knn import KNNClassifier as KDKNN


ITERS = 100
POINTS_COUNT = 1000
DIMENSIONALITY = 3
K = 6
METRIC = ("euclidean", 2)
KERNEL = None
H = 1.0
DATA_DISTRIBUTIONS = (
    "uniform",
    "groups_less_k",
    "groups_more_k",
)
GROUPS_LESS_KOEF = 0.5
GROUPS_MORE_KOEF = 2
BOUNDS = (-10, 10)
NOISE = 0.1
SEED = 24422442

def _distribution_name(distribution):
    names = {
        "uniform": "Разреженные",
        "groups_less_k": "Кучки меньше k",
        "groups_more_k": "Кучки больше k",
    }
    return names[distribution]

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

def _generate_groups_points(points_count, bounds, group_size, rng):
    groups_count = max(1, int(np.round(points_count / group_size)))
    centers = _generate_points(groups_count, bounds, rng)
    group_indices = np.arange(points_count) % groups_count

    X_train = _add_noise(centers[group_indices], NOISE, rng)
    return X_train


def _generate_data(points_count, bounds, distribution, k, rng):
    if distribution == "uniform":
        X_train = _generate_points(points_count, bounds, rng)
    elif distribution == "groups_less_k":
        group_size = max(2, int(k * GROUPS_LESS_KOEF))
        X_train = _generate_groups_points(points_count, bounds, group_size, rng)
    else:
        group_size = k * GROUPS_MORE_KOEF
        X_train = _generate_groups_points(points_count, bounds, group_size, rng)

    X_test = _generate_points(1, bounds, rng)
    X_test = _add_noise(X_test, NOISE, rng)

    return X_train, X_test

def _get_metrics(points_count=POINTS_COUNT, dimensionality=DIMENSIONALITY,
                 k=K, metric=METRIC, kernel=KERNEL, h=H,
                 distribution="uniform"):
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

        X_train, X_test = _generate_data(
            points_count,
            bounds,
            distribution,
            k,
            rng,
        )
        # чёт только сейчас понял, что оно не нужно в данном эксперименте...
        y_train = np.zeros(points_count, dtype=int)

        for name, classifier in (("base", BaseKNN), ("kd", KDKNN)):
            classi = classifier(
                k=k,
                distance_metric=distance_metric,
                p=p,
                kernel=kernel,
                h=h,
            )
            classi.fit(X_train, y_train)

            classi.predict(X_test)

            time, calculations = classi.metrics()
            metrics[f"{name}_time"].append(time)
            metrics[f"{name}_calculations"].append(calculations)

    return {
        name: float(np.mean(values))
        for name, values in metrics.items()
    }
