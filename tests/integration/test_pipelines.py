"""Flows Prefect de bout en bout sur des lots synthétiques (Prefect et MLflow temporaires)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import mlflow
import pandas as pd
import pytest
from prefect.server.api.server import create_app
from prefect.testing.utilities import prefect_test_harness
from sqlalchemy import Engine
from sqlalchemy.orm import Session
from starlette.testclient import TestClient

from bikeshare_mlops.config import get_settings
from bikeshare_mlops.models import registry
from bikeshare_mlops.pipelines.batch_prediction import batch_prediction
from bikeshare_mlops.pipelines.serve import watch_incoming
from bikeshare_mlops.pipelines.training import training_pipeline
from bikeshare_mlops.storage import repository
from factories import corrupt_frame, make_hourly_frame

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def _prefect() -> Iterator[None]:
    with prefect_test_harness(server_startup_timeout=120):
        yield


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, db_engine: Engine) -> Path:
    """Dossier du test (chemins isolés par conftest), base SQLite prête, forêt réduite."""
    monkeypatch.setenv("BIKESHARE_TRAINING__MODEL__N_ESTIMATORS", "10")
    monkeypatch.setenv("BIKESHARE_TRAINING__MODEL__N_JOBS", "1")
    monkeypatch.setenv("BIKESHARE_ORCHESTRATION__RETRY_DELAY_S", "0")
    return tmp_path


def _drop_months(directory: Path, frame: pd.DataFrame) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for month, chunk in frame.groupby(frame["dteday"].str.slice(0, 7)):
        chunk.to_csv(directory / f"batch_{month.replace('-', '_')}.csv", index=False)


def _last_run(engine: Engine) -> tuple[str, str | None, str | None]:
    with Session(engine) as session:
        run = repository.latest_pipeline_run(session)
    assert run is not None
    return run.state, run.outcome, run.model_version


def _alerts(workspace: Path) -> list[dict]:
    path = workspace / "logs_dir" / "alerts.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_prefect_server_dependencies_are_compatible() -> None:
    """Garde-fou de dépendances : le serveur Prefect doit accepter son client ET planifier des runs.

    Deux montées de version silencieuses (FastAPI >= 0.151, SQLAlchemy 2.1) ont déjà empêché
    toute planification sans faire échouer le démarrage. Le serveur est instancié dans ce
    processus : ce sont bien les versions installées ici qui sont testées.
    """
    with TestClient(create_app(ephemeral=True)) as client:
        # Requête de la CLI Prefect (champs à null) : 422 avec FastAPI >= 0.151.
        body = {"flows": None, "flow_runs": None, "limit": None, "offset": 0, "sort": None}
        response = client.post("/api/flows/filter", json=body)
        assert response.status_code == 200, response.text

        # Planification d'un déploiement à intervalle : échoue avec SQLAlchemy 2.1.
        flow_id = client.post("/api/flows/", json={"name": "compat-check"}).json()["id"]
        deployment = client.post(
            "/api/deployments/",
            json={
                "name": "compat-check",
                "flow_id": flow_id,
                "schedules": [{"schedule": {"interval": 60}, "active": True}],
            },
        )
        assert deployment.status_code in {200, 201}, deployment.text
        deployment_id = deployment.json()["id"]
        scheduled = client.post(f"/api/deployments/{deployment_id}/schedule", json={})
        assert scheduled.status_code in {200, 201}, scheduled.text
        runs = client.post(
            "/api/flow_runs/filter",
            json={"deployments": {"id": {"any_": [deployment_id]}}, "limit": 5},
        ).json()
        assert runs, "aucun run planifié"


def test_nominal_run_trains_registers_and_promotes(workspace: Path, db_engine: Engine) -> None:
    _drop_months(workspace / "incoming_dir", make_hourly_frame(days=59))

    summary = training_pipeline()

    assert summary["status"] == "promoted"
    assert summary["batches"] == ["batch_2011_01.csv", "batch_2011_02.csv"]
    assert not list((workspace / "incoming_dir").iterdir())
    assert len(list((workspace / "processed_dir").iterdir())) == 2
    settings = get_settings()
    assert registry.version_by_alias(settings, registry.CHAMPION) == summary["version"]
    assert _last_run(db_engine) == ("COMPLETED", "promoted", summary["version"])

    registry.configure_tracking(settings, settings.mlflow.ingestion_experiment_name)
    runs = mlflow.search_runs(experiment_names=[settings.mlflow.ingestion_experiment_name])
    assert set(runs["tags.validation_status"]) == {"passed"}
    assert len(runs) == 2

    # Prédiction par lot avec le champion, sur des données sans colonne cible.
    to_score = workspace / "to_score.csv"
    make_hourly_frame(start="2011-03-01", days=2).drop(
        columns=["cnt", "casual", "registered"]
    ).to_csv(to_score, index=False)
    result = batch_prediction(str(to_score))
    scored = pd.read_csv(result["output"])
    assert len(scored) == 48
    assert (scored["prediction"] >= 0).all()
    assert set(scored["model_version"].astype(str)) == {summary["version"]}


def test_invalid_batch_blocks_training_and_alerts(workspace: Path, db_engine: Engine) -> None:
    incoming = workspace / "incoming_dir"
    incoming.mkdir(parents=True)
    corrupt_frame(make_hourly_frame(days=31)).to_csv(incoming / "batch_invalid.csv", index=False)

    state = training_pipeline(return_state=True)

    assert state.is_failed()
    assert (workspace / "rejected_dir" / "batch_invalid.csv").exists()
    assert registry.version_by_alias(get_settings(), registry.CHAMPION) is None
    alert = _alerts(workspace)[-1]
    assert alert["event"] == "pipeline_failed"
    assert _last_run(db_engine)[0] == "FAILED"
    assert "batch_invalid" in alert["message"]


def test_valid_batches_wait_when_another_is_rejected(workspace: Path) -> None:
    incoming = workspace / "incoming_dir"
    _drop_months(incoming, make_hourly_frame(days=31))
    make_hourly_frame(days=31).drop(columns=["temp"]).to_csv(
        incoming / "batch_zz_bad.csv", index=False
    )

    assert training_pipeline(return_state=True).is_failed()
    assert [path.name for path in incoming.iterdir()] == ["batch_2011_01.csv"]


def test_nothing_to_do_without_new_batch(workspace: Path) -> None:
    assert training_pipeline() == {"status": "no_new_data"}
    assert watch_incoming() == {"status": "idle"}


def test_batch_prediction_rejects_invalid_input(workspace: Path) -> None:
    bad = workspace / "bad.csv"
    make_hourly_frame(days=1).assign(hr=99).to_csv(bad, index=False)

    state = batch_prediction(str(bad), return_state=True)

    assert state.is_failed()
    assert _alerts(workspace)[-1]["flow"] == "batch_prediction"
