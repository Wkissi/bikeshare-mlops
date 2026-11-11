"""API de bout en bout : TestClient, modèle factice, base SQLite (aucun service externe)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session

from bikeshare_mlops.api.main import create_app
from bikeshare_mlops.api.model_service import ModelService
from bikeshare_mlops.config import get_settings
from bikeshare_mlops.storage import repository
from bikeshare_mlops.storage.tables import Feedback, Prediction

pytestmark = pytest.mark.integration

VALID = {
    "season": 3, "yr": 1, "mnth": 7, "hr": 17, "holiday": 0, "weekday": 3, "workingday": 1,
    "weathersit": 1, "temp": 0.72, "atemp": 0.66, "hum": 0.45, "windspeed": 0.19,
}  # fmt: skip


class FakeModel:
    """Modèle factice déterministe : 10 locations par heure de la journée."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.fail:
            raise RuntimeError("panne du modèle")
        return frame["hr"].to_numpy(dtype=float) * 10


def _loader(version: str, model: FakeModel | None = None) -> Callable[[], tuple[Any, str]]:
    return lambda: (model or FakeModel(), version)


def _failing_loader() -> tuple[Any, str]:
    raise ConnectionError("registre MLflow injoignable")


@pytest.fixture
def make_client(db_engine: Engine) -> Iterator[Callable[..., TestClient]]:
    clients: list[TestClient] = []

    def _make(loader: Callable[[], tuple[Any, str]] | None = None) -> TestClient:
        service = ModelService("bikeshare-rf", "champion", loader or _loader("7"))
        # raise_server_exceptions=False : on observe la réponse 500 réellement renvoyée.
        app = create_app(get_settings(), service, db_engine)
        client = TestClient(app, raise_server_exceptions=False)
        client.__enter__()
        clients.append(client)
        return client

    yield _make
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client: Callable[..., TestClient]) -> TestClient:
    return make_client()


def _rows(engine: Engine) -> list[Prediction]:
    with Session(engine) as session:
        return list(session.scalars(select(Prediction)))


def test_predict_returns_traceable_prediction(client: TestClient, db_engine: Engine) -> None:
    response = client.post("/predict", json=VALID)

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == 170.0
    assert body["model_name"] == "bikeshare-rf"
    assert body["model_version"] == "7"
    row = _rows(db_engine)[0]
    assert row.id == body["prediction_id"]
    assert row.inputs == VALID
    assert row.status == "success"
    assert row.latency_ms >= 0


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({**VALID, "hr": 24}, "hr"),
        ({**VALID, "temp": 1.5}, "temp"),
        ({key: value for key, value in VALID.items() if key != "season"}, "season"),
        ({**VALID, "hr": "17"}, "hr"),
        ({**VALID, "hr": 17.5}, "hr"),
        ({**VALID, "cnt": 100}, "cnt"),
    ],
)
def test_invalid_input_returns_clear_422(
    client: TestClient, db_engine: Engine, payload: dict, field: str
) -> None:
    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "validation_error"
    assert field in body["message"]
    assert any(detail["field"] == field for detail in body["details"])
    assert _rows(db_engine) == []


def test_batch_prediction(client: TestClient, db_engine: Engine) -> None:
    instances = [{**VALID, "hr": hour} for hour in (8, 12, 18)]

    response = client.post("/predict/batch", json={"instances": instances})

    assert response.status_code == 200
    body = response.json()
    assert [item["prediction"] for item in body["predictions"]] == [80.0, 120.0, 180.0]
    assert {row.batch_id for row in _rows(db_engine)} == {body["batch_id"]}


@pytest.mark.parametrize("size", [0, 1001])
def test_batch_size_limits(client: TestClient, size: int) -> None:
    response = client.post("/predict/batch", json={"instances": [VALID] * size})
    assert response.status_code == 422
    assert "instances" in response.json()["message"]


def test_feedback_lifecycle(client: TestClient, db_engine: Engine) -> None:
    prediction_id = client.post("/predict", json=VALID).json()["prediction_id"]

    created = client.post("/feedback", json={"prediction_id": prediction_id, "actual": 200})
    duplicate = client.post("/feedback", json={"prediction_id": prediction_id, "actual": 150})

    assert created.status_code == 201
    assert created.json()["absolute_error"] == 30.0
    assert duplicate.status_code == 409
    assert duplicate.json()["error"] == "feedback_exists"
    with Session(db_engine) as session:
        assert repository.rolling_mae(session, 10) == (30.0, 1)
        assert session.scalar(select(Feedback.actual)) == 200


def test_feedback_errors(client: TestClient) -> None:
    unknown = client.post(
        "/feedback", json={"prediction_id": "00000000-0000-0000-0000-000000000000", "actual": 3}
    )
    malformed = client.post("/feedback", json={"prediction_id": "abc", "actual": 3})
    negative = client.post(
        "/feedback", json={"prediction_id": "00000000-0000-0000-0000-000000000000", "actual": -1}
    )

    assert unknown.status_code == 404
    assert unknown.json()["error"] == "prediction_not_found"
    assert malformed.status_code == 422
    assert negative.status_code == 422


def test_health_ok_and_model_info(client: TestClient) -> None:
    health = client.get("/health")
    model = client.get("/model")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["model"]["detail"] == "bikeshare-rf v7"
    assert model.json()["version"] == "7"
    assert client.get("/health/live").json() == {"status": "alive"}


def test_health_reports_last_pipeline_run(client: TestClient, db_engine: Engine) -> None:
    with Session(db_engine) as session:
        repository.record_pipeline_run(session, "training_pipeline", "r1", "COMPLETED", "promoted")

    assert client.get("/health").json()["last_pipeline_run"].startswith("COMPLETED (promoted)")


def test_startup_without_model_is_degraded(
    make_client: Callable[..., TestClient], db_engine: Engine
) -> None:
    client = make_client(_failing_loader)

    health = client.get("/health")
    prediction = client.post("/predict", json=VALID)

    assert health.status_code == 503
    assert health.json()["status"] == "degraded"
    assert "injoignable" in health.json()["model"]["detail"]
    assert prediction.status_code == 503
    assert prediction.json()["error"] == "model_unavailable"
    failed = _rows(db_engine)[0]
    assert failed.status == "error"
    assert "ModelUnavailableError" in failed.error

    feedback = client.post("/feedback", json={"prediction_id": failed.id, "actual": 3})
    assert feedback.status_code == 409


def test_database_down_gives_clean_503(client: TestClient, tmp_path: Path) -> None:
    broken = create_engine(f"sqlite:///{(tmp_path / 'absent' / 'x.db').as_posix()}")
    client.app.state.engine = broken
    client.app.state.sessions.configure(bind=broken)

    prediction = client.post("/predict", json=VALID)
    health = client.get("/health")

    assert prediction.status_code == 503
    assert prediction.json()["error"] == "database_unavailable"
    assert health.status_code == 503
    assert health.json()["database"]["ok"] is False


def test_unexpected_model_error_gives_500(
    make_client: Callable[..., TestClient], db_engine: Engine
) -> None:
    client = make_client(_loader("7", FakeModel(fail=True)))
    response = client.post("/predict", json=VALID)

    assert response.status_code == 500
    assert response.json() == {"error": "internal_error", "message": "Erreur interne inattendue."}
    assert "panne du modèle" in _rows(db_engine)[0].error


def test_reload_switches_version_or_keeps_current(make_client: Callable[..., TestClient]) -> None:
    versions = iter(["7", "8"])
    client = make_client(lambda: (FakeModel(), next(versions)))
    assert client.get("/model").json()["version"] == "7"

    assert client.post("/model/reload").json()["version"] == "8"

    client.app.state.model_service._loader = _failing_loader
    failed = client.post("/model/reload")
    assert failed.status_code == 503
    assert "La version 8 reste servie" in failed.json()["message"]
    assert client.post("/predict", json=VALID).json()["model_version"] == "8"
