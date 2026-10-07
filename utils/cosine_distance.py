import numpy as np

def cosine_similarity(x1, x2):
    dot_product = np.dot(x1, x2)
    norm_x1 = np.linalg.norm(x1)
    norm_x2 = np.linalg.norm(x2)

    if norm_x1 == 0 or norm_x2 == 0:
        return 0

    return dot_product / (norm_x1 * norm_x2)

def cosine_distance(x1, x2):
    return 1 - cosine_similarity(x1, x2)
