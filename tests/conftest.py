"""Fixtures partagées : configuration isolée dans un dossier temporaire, données synthétiques."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

from bikeshare_mlops.config import PathsConfig, Settings, get_settings
from factories import make_hourly_frame


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> Iterator[None]:
    """Vide le cache de configuration avant et après chaque test."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _isolate_mlflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Garde-fou : aucun test ne peut écrire dans le vrai serveur MLflow.

    Par défaut, MLflow pointe vers une base SQLite jetable propre à chaque test.
    """
    monkeypatch.setenv(
        "BIKESHARE_MLFLOW__TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    )
    monkeypatch.setenv("BIKESHARE_MLFLOW__ARTIFACT_LOCATION", (tmp_path / "artifacts").as_uri())
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Configuration du dépôt dont tous les chemins pointent vers un dossier temporaire."""
    base = get_settings()
    data = tmp_path / "data"
    paths = PathsConfig(
        data_dir=data,
        raw_dir=data / "raw",
        batches_dir=data / "batches",
        incoming_dir=data / "incoming",
        processed_dir=data / "processed",
        rejected_dir=data / "rejected",
        reports_dir=tmp_path / "reports",
        logs_dir=tmp_path / "logs",
    )
    return base.model_copy(update={"paths": paths})


@pytest.fixture
def hourly_frame() -> pd.DataFrame:
    """Un mois synthétique conforme au schéma (744 lignes)."""
    return make_hourly_frame()
