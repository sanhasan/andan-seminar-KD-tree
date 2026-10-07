import numpy as np

def minkowski_distance(x1, x2, p=2):
    if p == np.inf:
        return np.max(np.abs(x1 - x2))
    else:
        return np.sum(np.abs(x1 - x2) ** p) ** (1/p)
