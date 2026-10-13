from benchmarks.common_benchmark import (
    DATA_DISTRIBUTIONS,
    _distribution_name,
    _get_metrics,
)


K = (3, 5, 10, 20, 50, 100, 200, 500)


def run():
    result = []
    for distribution in DATA_DISTRIBUTIONS:
        for k in K:
            metrics = _get_metrics(k=k, distribution=distribution)
            metrics["k"] = k
            metrics["distribution"] = _distribution_name(distribution)
            result.append(metrics)
    return result
