"""Prétraitement sklearn, sérialisé AVEC le modèle dans un seul ``Pipeline``.

Le même objet transforme les données à l'entraînement et à l'inférence : impossible que l'API
applique un encodage différent de celui vu pendant l'apprentissage (« training-serving skew »).

``casual`` et ``registered`` sont exclues : leur somme vaut exactement la cible ``cnt`` (fuite).
``instant`` (numéro de ligne) et ``dteday`` (déjà décomposée en yr/mnth/weekday) le sont aussi.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from bikeshare_mlops.config import ModelParams

TARGET = "cnt"
CYCLICAL_PERIODS = {"hr": 24, "mnth": 12, "weekday": 7}
CATEGORICAL = ["season", "weathersit"]
PASSTHROUGH = ["yr", "holiday", "workingday", "temp", "atemp", "hum", "windspeed"]
FEATURES = [*CYCLICAL_PERIODS, *CATEGORICAL, *PASSTHROUGH]


class CyclicalEncoder(TransformerMixin, BaseEstimator):
    """Encode une variable périodique en (sin, cos) : 23 h et 0 h deviennent voisines."""

    def __init__(self, period: float) -> None:
        self.period = period

    def fit(self, features: pd.DataFrame, target: object = None) -> CyclicalEncoder:
        self.feature_names_in_ = np.asarray(features.columns, dtype=object)
        return self

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        angles = 2 * np.pi * np.asarray(features, dtype=float) / self.period
        return np.hstack([np.sin(angles), np.cos(angles)])

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        names = list(self.feature_names_in_)
        return np.asarray([f"{n}_sin" for n in names] + [f"{n}_cos" for n in names], dtype=object)


def build_preprocessor() -> ColumnTransformer:
    """Transformation des colonnes brutes en variables numériques pour le modèle."""
    cyclical = [
        (f"{column}_cyclic", CyclicalEncoder(period), [column])
        for column, period in CYCLICAL_PERIODS.items()
    ]
    return ColumnTransformer(
        [
            *cyclical,
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL,
            ),
            ("passthrough", "passthrough", PASSTHROUGH),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(params: ModelParams) -> Pipeline:
    """Pipeline complet prétraitement + forêt aléatoire."""
    return Pipeline(
        [
            ("preprocess", build_preprocessor()),
            ("model", RandomForestRegressor(**params.model_dump())),
        ]
    )


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Sépare les variables explicatives (colonnes FEATURES) de la cible."""
    return frame[FEATURES], frame[TARGET]
