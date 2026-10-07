from benchmarks.common_benchmark import _get_metrics


K = (1, 3, 5, 10, 20, 50, 100, 200, 500)


def run():
    result = []
    for k in K:
        metrics = _get_metrics(k=k)
        metrics["k"] = k
        result.append(metrics)
    return result
