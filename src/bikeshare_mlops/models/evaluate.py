"""Évaluation : métriques globales et par segment, graphiques de diagnostic."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # rendu sans écran (serveur, CI, conteneur)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

SEGMENT_COLUMN = "season"


@dataclass(frozen=True)
class Metrics:
    """Erreurs de régression : RMSE pénalise les grosses erreurs, MAE est en « vélos/heure »."""

    rmse: float
    mae: float
    r2: float

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        return {f"{prefix}{name}": value for name, value in asdict(self).items()}


def compute_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> Metrics:
    """RMSE, MAE et R² d'une série de prédictions."""
    return Metrics(
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        mae=float(mean_absolute_error(y_true, y_pred)),
        r2=float(r2_score(y_true, y_pred)),
    )


def metrics_by_segment(
    frame: pd.DataFrame, y_pred: np.ndarray, target: str, column: str = SEGMENT_COLUMN
) -> dict[str, Metrics]:
    """Métriques par valeur de ``column`` (sert aux tests de non-régression par segment)."""
    results: dict[str, Metrics] = {}
    predictions = pd.Series(y_pred, index=frame.index)
    for value, group in frame.groupby(column):
        group_pred = predictions.loc[group.index].to_numpy()
        results[str(value)] = compute_metrics(group[target], group_pred)
    return results


def feature_importances(pipeline: Pipeline) -> pd.Series:
    """Importances du modèle, nommées d'après les sorties du prétraitement."""
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    values = pipeline.named_steps["model"].feature_importances_
    return pd.Series(values, index=names).sort_values(ascending=False)


def save_diagnostic_plots(
    test_frame: pd.DataFrame,
    y_pred: np.ndarray,
    target: str,
    importances: pd.Series,
    output_dir: Path,
) -> list[Path]:
    """Prédit vs réel, résidus, erreur par heure et importances : 4 PNG dans ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    y_true = test_frame[target].to_numpy()
    residuals = y_true - y_pred
    paths: list[Path] = []

    def _save(fig: plt.Figure, name: str) -> None:
        path = output_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        paths.append(path)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true, y_pred, s=4, alpha=0.3)
    upper = float(max(y_true.max(), y_pred.max()))
    ax.plot([0, upper], [0, upper], color="black", linewidth=1)
    ax.set(xlabel="Locations réelles", ylabel="Locations prédites", title="Prédit vs réel")
    _save(fig, "predicted_vs_actual.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residuals, bins=60)
    ax.set(xlabel="Résidu (réel - prédit)", ylabel="Effectif", title="Distribution des résidus")
    _save(fig, "residuals.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    abs_error = pd.Series(np.abs(residuals)).groupby(test_frame["hr"].to_numpy()).mean()
    ax.bar(abs_error.index, abs_error.to_numpy())
    ax.set(xlabel="Heure", ylabel="MAE", title="Erreur absolue moyenne par heure")
    _save(fig, "error_by_hour.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    top = importances.head(15).iloc[::-1]
    ax.barh(top.index, top.to_numpy())
    ax.set(xlabel="Importance", title="Importance des variables (top 15)")
    _save(fig, "feature_importances.png")
    return paths
