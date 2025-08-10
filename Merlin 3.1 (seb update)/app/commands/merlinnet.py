import numpy as np
from typing import Optional

class MerlinNetClassifier:
    """
    A tiny from-scratch classifier (1 hidden layer, ReLU, sigmoid output).
    API mimics scikit-learn (fit, predict_proba, predict).
    """
    def __init__(self, hidden=32, lr=0.01, epochs=200, batch_size=256, seed=42, l2=1e-4):
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.l2 = l2
        self.W1 = self.b1 = self.W2 = self.b2 = None
        self._fitted = False

    def _init_weights(self, in_dim):
        rng = np.random.default_rng(self.seed)
        limit = np.sqrt(6/(in_dim + self.hidden))
        self.W1 = rng.uniform(-limit, limit, size=(in_dim, self.hidden))
        self.b1 = np.zeros((1, self.hidden))
        self.W2 = rng.uniform(-np.sqrt(6/(self.hidden+1)), np.sqrt(6/(self.hidden+1)), size=(self.hidden, 1))
        self.b2 = np.zeros((1, 1))

    @staticmethod
    def _relu(x):
        return np.maximum(0, x)

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def _forward(self, X):
        Z1 = X @ self.W1 + self.b1
        A1 = self._relu(Z1)
        Z2 = A1 @ self.W2 + self.b2
        A2 = self._sigmoid(Z2)
        cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
        return A2, cache

    def _backward(self, y, cache):
        X, A1, A2 = cache["X"], cache["A1"], cache["A2"]
        m = X.shape[0]
        dZ2 = (A2 - y.reshape(-1,1)) / m
        dW2 = A1.T @ dZ2 + self.l2 * self.W2
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (cache["Z1"] > 0)
        dW1 = X.T @ dZ1 + self.l2 * self.W1
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        return dW1, db1, dW2, db2

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        if self.W1 is None:
            self._init_weights(d)
        idx = np.arange(n)
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.batch_size):
                batch = idx[start:start+self.batch_size]
                A2, cache = self._forward(X[batch])
                dW1, db1, dW2, db2 = self._backward(y[batch], cache)
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1
                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2
        self._fitted = True
        return self

    def predict_proba(self, X):
        if not self._fitted:
            raise RuntimeError("MerlinNet not fitted.")
        X = np.asarray(X, dtype=float)
        A2, _ = self._forward(X)
        # return shape (n,2) like sklearn: [P0, P1]
        return np.hstack([1 - A2, A2])

    def predict(self, X):
        return (self.predict_proba(X)[:,1] >= 0.5).astype(int)