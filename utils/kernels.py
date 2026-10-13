import numpy as np


def uniform_kernel(u):
    return 0.5 if np.abs(u) < 1 else 0


def triangular_kernel(u):
    return 1 - np.abs(u) if np.abs(u) < 1 else 0


def epanechnikov_kernel(u):
    return 0.75 * (1 - u ** 2) if np.abs(u) < 1 else 0


def gaussian_kernel(u):
    return np.exp(-0.5 * u ** 2) / np.sqrt(2 * np.pi)


KERNELS = {
    "uniform": uniform_kernel,
    "triangular": triangular_kernel,
    "epanechnikov": epanechnikov_kernel,
    "gaussian": gaussian_kernel,
}
