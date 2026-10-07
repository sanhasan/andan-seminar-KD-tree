from heapq import heappush, heapreplace

import numpy as np

from utils.knnClassifier import KNNClassifier as AbstractKNNClassifier
from utils.minkowski_distance import minkowski_distance


class KNNClassifier(AbstractKNNClassifier):
    def __init__(self, k=5, distance_metric='euclidean', p=2):
        super().__init__(k=k, distance_metric=distance_metric, p=p)
        self._root = None

    def fit(self, X, y):
        super().fit(X, y)
        self._root = self._build_tree(np.arange(len(self.X_train)), 0)
        return self

    def _find_dists(self, x):
        distances = []
        self._search(x, self._root, distances)
        return sorted((-distance, -index) for distance, index in distances)

    def _build_tree(self, indices, depth):
        if len(indices) == 0:
            return None
        
        points = self.X_train[indices]
        lower = np.min(points, axis=0)
        upper = np.max(points, axis=0)

        axis = depth % self.X_train.shape[1]
        order = np.argsort(self.X_train[indices, axis], kind="stable")
        indices = indices[order]

        middle = len(indices) // 2
        index = int(indices[middle])

        left = self._build_tree(indices[:middle], depth + 1)
        right = self._build_tree(indices[middle + 1:], depth + 1)

        return index, axis, left, right, lower, upper

    def _search(self, x, node, kbest):
        if node is None:
            return

        index, axis, left, right, lower, upper = node

        if len(kbest) == self.k and self.distance_metric != 'cosine':
            p = self.p
            if self.distance_metric == 'euclidean':
                p = 2
            elif self.distance_metric == 'manhattan':
                p = 1
            
            nearest = np.clip(x, lower, upper)
            if minkowski_distance(x, nearest, p=p) > -kbest[0][0]:
                return

        tmp = self.X_train[index]
        dist = (-self._compute_distance(x, tmp), -index)

        if len(kbest) < self.k:
            heappush(kbest, dist)
        elif dist > kbest[0]:
            heapreplace(kbest, dist)

        if x[axis] < tmp[axis]:
            near, far = left, right
        else:
            near, far = right, left
        self._search(x, near, kbest)
        self._search(x, far, kbest)
