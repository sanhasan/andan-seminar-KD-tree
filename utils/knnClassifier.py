from abc import ABC, abstractmethod
from time import perf_counter

import numpy as np

from utils.cosine_distance import cosine_distance
from utils.minkowski_distance import minkowski_distance


class KNNClassifier(ABC):
    def __init__(self, k=5, distance_metric='euclidean', p=2):
        self.k = k
        self.distance_metric = distance_metric
        self.p = p
        self.X_train = None
        self.y_train = None
        self._time = 0.0
        self._cnt_calculations = 0

    def _compute_distance(self, x1, x2):
        self._cnt_calculations += 1

        if self.distance_metric == 'euclidean':
            return minkowski_distance(x1, x2, p=2)
        elif self.distance_metric == 'manhattan':
            return minkowski_distance(x1, x2, p=1)
        elif self.distance_metric == 'minkowski':
            return minkowski_distance(x1, x2, p=self.p)
        elif self.distance_metric == 'cosine':
            return cosine_distance(x1, x2)

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self

    @abstractmethod
    def _find_dists(self, x):
        pass

    def predict(self, X):
        start = perf_counter()
        self._cnt_calculations = 0

        predictions = []
        for x in X:
            distances = self._find_dists(x)

            k_nearest_indices = [idx for _, idx in distances[:self.k]]
            k_nearest_labels = self.y_train[k_nearest_indices]

            unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
            predicted_label = unique_labels[np.argmax(counts)]
            predictions.append(predicted_label)

        self._time = perf_counter() - start

        return np.array(predictions)

    def predict_proba(self, X):
        start = perf_counter()
        self._cnt_calculations = 0

        probabilities = []
        for x in X:
            distances = self._find_dists(x)
            
            k_nearest_indices = [idx for _, idx in distances[:self.k]]
            k_nearest_labels = self.y_train[k_nearest_indices]

            unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
            proba = np.zeros(len(unique_labels))
            for i, label in enumerate(unique_labels):
                proba[i] = np.sum(k_nearest_labels == label) / self.k

            probabilities.append(dict(zip(unique_labels, proba)))

        self._time = perf_counter() - start

        return probabilities

    def metrics(self):
        return self._time, self._cnt_calculations
