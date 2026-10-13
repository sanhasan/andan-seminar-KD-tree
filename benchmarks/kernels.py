from benchmarks.common_benchmark import (
    DATA_DISTRIBUTIONS,
    _distribution_name,
    _get_metrics,
)


KERNEL = (
    "uniform",
    "triangular",
    "epanechnikov",
    "gaussian",
)
H = (0.1, 0.5, 1.0, 5.0, 10.0)


def _kernel_name(kernel):
    names = {
        "uniform": "Равномерное",
        "triangular": "Треугольное",
        "epanechnikov": "Епанечникова",
        "gaussian": "Гауссово",
    }
    return names[kernel]


def run():
    result = []
    for distribution in DATA_DISTRIBUTIONS:
        for kernel in KERNEL:
            for h in H:
                metrics = _get_metrics(
                    kernel=kernel,
                    h=h,
                    distribution=distribution,
                )
                metrics["h"] = h
                metrics["kernel"] = _kernel_name(kernel)
                metrics["distribution"] = _distribution_name(distribution)
                result.append(metrics)
    return result
