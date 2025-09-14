import numpy as np

class KNNRegressor:
    def __init__(self, k=5, distance="euclidean"):
        self.k = int(k)
        self.distance = distance
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)
        return self

    def _pairwise_dist(self, A, B):
        if self.distance != "euclidean":
            raise ValueError("Only euclidean distance implemented in this simple version.")
        # Efficient L2 distances via (a-b)^2 = a^2 + b^2 - 2ab
        A2 = np.sum(A*A, axis=1, keepdims=True)          # [nA, 1]
        B2 = np.sum(B*B, axis=1, keepdims=True).T        # [1, nB]
        D2 = A2 + B2 - 2.0 * (A @ B.T)                   # [nA, nB]
        np.maximum(D2, 0, out=D2)                        # numerical safety
        return np.sqrt(D2)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        D = self._pairwise_dist(X, self.X)               # [n_samples, n_train]
        # take indices of k smallest distances
        nn_idx = np.argpartition(D, kth=self.k-1, axis=1)[:, :self.k]
        preds = np.mean(self.y[nn_idx], axis=1)
        return preds
