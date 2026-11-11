"""Fixtures partagées : configuration isolée dans un dossier temporaire, données synthétiques."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import Engine

from bikeshare_mlops.config import PathsConfig, Settings, get_settings
from bikeshare_mlops.storage.db import build_engine
from bikeshare_mlops.storage.tables import Base
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


@pytest.fixture(autouse=True)
def _isolate_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Garde-fou : par défaut, tous les chemins de la config pointent dans le dossier du test
    (aucun test ne lit ni n'écrit les vrais data/, reports/ ou logs/ du projet)."""
    for key in PathsConfig.model_fields:
        monkeypatch.setenv(f"BIKESHARE_PATHS__{key.upper()}", str(tmp_path / key))


@pytest.fixture(autouse=True)
def _isolate_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Garde-fou : aucun test n'écrit dans le PostgreSQL réel (SQLite jetable par test)."""
    monkeypatch.setenv("BIKESHARE_DATABASE__URL", f"sqlite:///{(tmp_path / 'app.db').as_posix()}")


@pytest.fixture
def db_engine(_isolate_database: None) -> Iterator[Engine]:
    """Base SQLite du test avec le schéma applicatif créé.

    La configuration est lue sans passer par le cache : une fixture appelée ensuite peut encore
    surcharger d'autres variables d'environnement.
    """
    engine = build_engine(Settings())
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


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
