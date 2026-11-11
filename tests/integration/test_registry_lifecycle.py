"""Cycle de vie complet dans un registre MLflow local (SQLite temporaire, aucun serveur) :
entraînement → challenger → promotion → rejet → rollback.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import pytest

from bikeshare_mlops.config import MlflowConfig, ModelParams, Settings
from bikeshare_mlops.data.ingest import BatchInfo
from bikeshare_mlops.exceptions import RegistryError
from bikeshare_mlops.models import registry
from bikeshare_mlops.models.promote import evaluate_and_promote, rollback
from bikeshare_mlops.models.train import log_and_register, train_and_evaluate
from factories import make_hourly_frame

pytestmark = pytest.mark.integration


@pytest.fixture
def mlflow_settings(settings: Settings, tmp_path: Path) -> Settings:
    tracking = MlflowConfig(
        tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        experiment_name="test-exp",
        registered_model_name="test-model",
        artifact_location=(tmp_path / "artifacts").as_uri(),
    )
    training = settings.training.model_copy(
        update={"model": ModelParams(n_estimators=10, n_jobs=1)}
    )
    return settings.model_copy(update={"mlflow": tracking, "training": training})


def _batch() -> BatchInfo:
    return BatchInfo("batch_2011_01", Path("batch_2011_01.csv"), 744, "a" * 64, "", "", "")


def _status(settings: Settings, version: str) -> str:
    entry = registry.client(settings).get_model_version("test-model", version)
    return entry.tags[registry.TAG_STATUS]


def test_full_lifecycle(mlflow_settings: Settings, tmp_path: Path) -> None:
    frame = make_hourly_frame(days=120)

    # 1. Premier entraînement : tracé, enregistré en challenger, puis promu (pas de champion).
    result = train_and_evaluate(frame, mlflow_settings, tmp_path / "run1")
    first = log_and_register(result, mlflow_settings, [_batch()])
    assert registry.version_by_alias(mlflow_settings, registry.CHALLENGER) == first.version

    run = mlflow.get_run(first.run_id)
    assert run.data.tags["data_sha256"] == result.data_sha256
    assert run.data.metrics["test_rmse"] == pytest.approx(result.metrics.rmse)
    assert "model.n_estimators" in run.data.params
    assert run.data.tags["code_version"]

    decision = evaluate_and_promote(mlflow_settings, frame)
    assert decision.promoted
    assert registry.version_by_alias(mlflow_settings, registry.CHAMPION) == first.version
    assert registry.version_by_alias(mlflow_settings, registry.CHALLENGER) is None

    # 2. Même données, même graine : aucun gain → rejeté avec un tag explicite.
    second = log_and_register(
        train_and_evaluate(frame, mlflow_settings, tmp_path / "run2"), mlflow_settings, [_batch()]
    )
    rejected = evaluate_and_promote(mlflow_settings, frame)
    assert not rejected.promoted
    assert _status(mlflow_settings, second.version) == "rejected"
    assert registry.version_by_alias(mlflow_settings, registry.CHAMPION) == first.version

    # 3. Rollback : le champion pointe sur la version demandée, l'ancien est marqué.
    previous = rollback(mlflow_settings, second.version)
    assert previous == first.version
    assert registry.version_by_alias(mlflow_settings, registry.CHAMPION) == second.version
    assert _status(mlflow_settings, first.version) == "rolled_back"

    with pytest.raises(RegistryError):
        rollback(mlflow_settings, second.version)
    with pytest.raises(RegistryError):
        rollback(mlflow_settings, "99")

    rows = registry.list_versions(mlflow_settings)
    assert [row.version for row in rows] == [second.version, first.version]
    assert rows[0].aliases == "champion"


def test_promotion_without_challenger_fails(mlflow_settings: Settings) -> None:
    registry.configure_tracking(mlflow_settings)
    with pytest.raises(RegistryError):
        evaluate_and_promote(mlflow_settings, make_hourly_frame(days=60))
