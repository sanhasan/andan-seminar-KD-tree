from benchmarks.common_benchmark import _get_metrics

POINTS_COUNT = (10, 20, 50, 100, 200, 500, 1000, 2000, 5000)

def run():
    result = []
    for points_count in POINTS_COUNT:
        metrics = _get_metrics(points_count=points_count)
        metrics["points_count"] = points_count
        result.append(metrics)
    return result
