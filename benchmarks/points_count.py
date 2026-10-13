from benchmarks.common_benchmark import (
    DATA_DISTRIBUTIONS,
    _distribution_name,
    _get_metrics,
)

POINTS_COUNT = (10, 20, 50, 100, 200, 500, 1000, 2000, 5000)

def run():
    result = []
    for distribution in DATA_DISTRIBUTIONS:
        for points_count in POINTS_COUNT:
            metrics = _get_metrics(
                points_count=points_count,
                distribution=distribution,
            )
            metrics["points_count"] = points_count
            metrics["distribution"] = _distribution_name(distribution)
            result.append(metrics)
    return result
