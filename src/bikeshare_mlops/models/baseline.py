"""Baseline : moyenne historique des locations par (heure, jour ouvré).

Elle fixe le niveau « sans apprentissage » : un modèle qui ne la bat pas nettement n'apporte rien.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

GROUP_KEYS = ["hr", "workingday"]


class HourlyMeanBaseline(RegressorMixin, BaseEstimator):
    """Prédit la moyenne observée pour la même heure et le même type de jour."""

    def fit(self, features: pd.DataFrame, target: pd.Series) -> HourlyMeanBaseline:
        grouped = target.groupby([features[key] for key in GROUP_KEYS]).mean()
        self.means_ = grouped.to_dict()
        self.global_mean_ = float(target.mean())
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        keys = zip(*(features[key] for key in GROUP_KEYS), strict=True)
        return np.array([self.means_.get(key, self.global_mean_) for key in keys], dtype=float)
