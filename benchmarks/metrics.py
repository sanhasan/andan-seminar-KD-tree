import numpy as np

from benchmarks.common_benchmark import _get_metrics


METRIC = (
    ("manhattan", 1),
    ("minkowski", 1.5),
    ("euclidean", 2),
    ("minkowski", 3),
    ("minkowski", 5),
    ("minkowski", 10),
    ("minkowski", 20),
    ("minkowski", 50),
    ("minkowski", np.inf),
    ("cosine", 2),
)


def _metric_name(metric):
    distance_metric, p = metric
    if distance_metric == "manhattan":
        return "Манхэттенская"
    if distance_metric == "euclidean":
        return "Евклидова"
    if distance_metric == "cosine":
        return "Косинусная"
    exponent = "inf" if p == np.inf else f"{p:g}"
    return f"Минковского, p={exponent}"


def run():
    result = []
    for metric in METRIC:
        metrics = _get_metrics(metric=metric)
        metrics["metric"] = _metric_name(metric)
        result.append(metrics)
    return result
