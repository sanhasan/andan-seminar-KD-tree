from benchmarks.common_benchmark import _get_metrics


DIMENSIONALITY = (1, 2, 3, 5, 10, 15, 20, 30, 50, 100)


def run():
    result = []
    for dimensionality in DIMENSIONALITY:
        metrics = _get_metrics(dimensionality=dimensionality)
        metrics["dimensionality"] = dimensionality
        result.append(metrics)
    return result
