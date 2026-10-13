from benchmarks.common_benchmark import (
    DATA_DISTRIBUTIONS,
    _distribution_name,
    _get_metrics,
)


DIMENSIONALITY = (1, 2, 3, 5, 10, 15, 20, 30, 50, 100)


def run():
    result = []
    for distribution in DATA_DISTRIBUTIONS:
        for dimensionality in DIMENSIONALITY:
            metrics = _get_metrics(
                dimensionality=dimensionality,
                distribution=distribution,
            )
            metrics["dimensionality"] = dimensionality
            metrics["distribution"] = _distribution_name(distribution)
            result.append(metrics)
    return result
