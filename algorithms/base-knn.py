from utils.knnClassifier import KNNClassifier as AbstractKNNClassifier


class KNNClassifier(AbstractKNNClassifier):
    def _find_dists(self, x):
        distances = []
        for i, train_point in enumerate(self.X_train):
            dist = self._compute_distance(x, train_point)
            distances.append((dist, i))

        distances.sort(key=lambda x: x[0])
        return distances
