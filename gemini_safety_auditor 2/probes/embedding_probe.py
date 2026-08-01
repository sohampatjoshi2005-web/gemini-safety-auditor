"""
RECONSTRUCTED FILE -- not part of your original upload.

Rebuilt from the interface your other files assume:
  - smoke_test.py: `probe = EmbeddingProbe(); probe.fit(embeddings, labels)`
  - cross_tier.py / multimodal_probe.py: `probe.score(embed(text))` returns
    something numeric, stored as `probe_unsafe_score: float | None`.
  - README.md: "a linear probe trained on Gemini's embeddings" /
    "logistic-regression probe on embeddings".

requirements.txt does not list scikit-learn, so this implements plain
logistic regression by hand with numpy (gradient descent on the log-loss)
rather than assuming an sklearn dependency you didn't declare. If your
original used scikit-learn (or a different scoring convention), swap this
file for the real one.

score() returns the model's estimated P(unsafe) for a given embedding, in
[0, 1]. Callers generally treat > 0.5 as "probe thinks this looks unsafe."
"""

from __future__ import annotations

import numpy as np


class EmbeddingProbe:
    def __init__(self, lr: float = 0.1, epochs: int = 500, l2: float = 1e-3):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, embeddings: list[list[float]], labels: list[int]) -> "EmbeddingProbe":
        X = np.asarray(embeddings, dtype=float)
        y = np.asarray(labels, dtype=float)

        # standardize features so gradient descent behaves regardless of the
        # raw embedding model's scale
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0)
        self._std[self._std == 0] = 1.0
        Xs = (X - self._mean) / self._std

        n_features = Xs.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        n = Xs.shape[0]

        for _ in range(self.epochs):
            z = Xs @ self.weights + self.bias
            preds = self._sigmoid(z)
            error = preds - y
            grad_w = (Xs.T @ error) / n + self.l2 * self.weights
            grad_b = error.mean()
            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b

        return self

    def score(self, embedding: list[float]) -> float:
        """Returns P(unsafe) in [0, 1] for a single embedding."""
        if self.weights is None:
            raise RuntimeError("EmbeddingProbe.fit() must be called before score().")
        x = (np.asarray(embedding, dtype=float) - self._mean) / self._std
        z = float(x @ self.weights + self.bias)
        return float(self._sigmoid(np.array([z]))[0])

    def predict_unsafe(self, embedding: list[float], threshold: float = 0.5) -> bool:
        return self.score(embedding) >= threshold
