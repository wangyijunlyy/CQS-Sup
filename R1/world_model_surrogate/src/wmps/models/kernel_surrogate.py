"""h_cs-style kernel surrogate: truncated trigonometric features + ridge."""

from __future__ import annotations

from itertools import combinations, product

import numpy as np


def build_trig_features(X: np.ndarray, Lambda: int) -> np.ndarray:
    """Truncated trigonometric monomial features Phi_omega(x).

    Phi_omega(x) = prod_i [1_{omega_i=0} + 1_{omega_i=1} cos(x_i)
                           + 1_{omega_i=-1} sin(x_i)],  ||omega||_0 <= Lambda.
    Returns [n, m].
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n, d = X.shape
    cols = [np.ones((n, 1))]
    for k in range(1, Lambda + 1):
        for idx in combinations(range(d), k):
            for signs in product([1, -1], repeat=k):
                f = np.ones(n)
                for j, s in zip(idx, signs):
                    f = f * (np.cos(X[:, j]) if s == 1 else np.sin(X[:, j]))
                cols.append(f.reshape(-1, 1))
    return np.hstack(cols)


class KernelSurrogate:
    """Ridge regression on truncated trigonometric features (paper h_cs-like)."""

    def __init__(self, Lambda: int = 2, lam: float = 1.0, seed: int = 42):
        self.Lambda = Lambda
        self.lam = lam
        self.seed = seed
        self.coef_ = None
        self.feature_dim_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KernelSurrogate":
        X = np.atleast_2d(X)
        y = np.asarray(y, dtype=float)
        Phi = build_trig_features(X, self.Lambda)
        self.feature_dim_ = Phi.shape[1]
        Gram = Phi.T @ Phi
        reg = self.lam * np.eye(Gram.shape[0])
        self.coef_ = np.linalg.solve(Gram + reg, Phi.T @ y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("call fit() first")
        Phi = build_trig_features(X, self.Lambda)
        return Phi @ self.coef_
