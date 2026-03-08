from __future__ import annotations

from typing import Union

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class LDA:
    """Thin wrapper around sklearn LDA for consistent imports."""

    def __init__(
        self, solver: str = "lsqr", shrinkage: Union[str, float, None] = "auto"
    ):
        self.model = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)

    def fit(self, X: np.ndarray, y: np.ndarray):  # noqa: N803
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        return self.model.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:  # noqa: N803
        return float(self.model.score(X, y))
