"""Exposition Prometheus : trafic, métriques métier et état calculé depuis la base."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from bikeshare_mlops.api.main import create_app
from bikeshare_mlops.api.model_service import ModelService
from bikeshare_mlops.config import get_settings
from bikeshare_mlops.monitoring.metrics import record_served_version
from bikeshare_mlops.storage import repository

pytestmark = pytest.mark.integration

VALID = {
    "season": 3, "yr": 1, "mnth": 7, "hr": 17, "holiday": 0, "weekday": 3, "workingday": 1,
    "weathersit": 1, "temp": 0.72, "atemp": 0.66, "hum": 0.45, "windspeed": 0.19,
}  # fmt: skip


class _HourModel:
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame["hr"].to_numpy(dtype=float) * 10


@pytest.fixture
def client(db_engine: Engine) -> Iterator[TestClient]:
    loader: Callable[[], tuple[Any, str]] = lambda: (_HourModel(), "42")  # noqa: E731
    service = ModelService("bikeshare-rf", "champion", loader, on_loaded=record_served_version)
    with TestClient(create_app(get_settings(), service, db_engine)) as test_client:
        yield test_client


def _samples(client: TestClient) -> dict[tuple[str, tuple[tuple[str, str], ...]], float]:
    response = client.get("/metrics")
    assert response.status_code == 200
    samples = {}
    for family in text_string_to_metric_families(response.text):
        for sample in family.samples:
            samples[(sample.name, tuple(sorted(sample.labels.items())))] = sample.value
    return samples


def _value(samples: dict, name: str, **labels: str) -> float:
    return samples.get((name, tuple(sorted(labels.items()))), 0.0)


def test_traffic_and_business_metrics(client: TestClient, db_engine: Engine) -> None:
    before = _samples(client)
    prediction_id = client.post("/predict", json=VALID).json()["prediction_id"]
    client.post("/predict/batch", json={"instances": [VALID, VALID]})
    client.post("/feedback", json={"prediction_id": prediction_id, "actual": 150})
    client.post("/predict", json={**VALID, "hr": 30})
    with Session(db_engine) as session:
        repository.record_pipeline_run(session, "training_pipeline", "r1", "FAILED")

    after = _samples(client)

    def delta(name: str, **labels: str) -> float:
        return _value(after, name, **labels) - _value(before, name, **labels)

    assert delta("http_requests_total", handler="/predict", method="POST", status="200") == 1
    assert delta("http_requests_total", handler="/predict", method="POST", status="422") == 1
    assert (
        delta(
            "bikeshare_predictions_total", model_version="42", status="success", endpoint="single"
        )
        == 1
    )
    assert (
        delta("bikeshare_predictions_total", model_version="42", status="success", endpoint="batch")
        == 2
    )
    assert delta("bikeshare_feedbacks_total") == 1
    assert delta("bikeshare_api_errors_total", error="validation_error") == 1
    assert _value(after, "bikeshare_served_model_version") == 42
    assert _value(after, "bikeshare_rolling_mae") == 20.0
    assert _value(after, "bikeshare_rolling_mae_samples") == 1
    assert _value(after, "bikeshare_database_up") == 1
    assert _value(after, "bikeshare_pipeline_runs", state="FAILED") == 1
    assert _value(after, "bikeshare_last_pipeline_run_success") == 0
    assert any(name == "http_request_duration_seconds_bucket" for name, _ in after)


def test_database_down_is_visible(client: TestClient, tmp_path: Path) -> None:
    broken = create_engine(f"sqlite:///{(tmp_path / 'absent' / 'x.db').as_posix()}")
    client.app.state.engine = broken
    client.app.state.sessions.configure(bind=broken)

    before = _samples(client)
    client.post("/predict", json=VALID)
    after = _samples(client)

    assert _value(after, "bikeshare_database_up") == 0
    assert (
        _value(after, "bikeshare_api_errors_total", error="database_unavailable")
        - _value(before, "bikeshare_api_errors_total", error="database_unavailable")
    ) == 1
    assert ("bikeshare_rolling_mae", ()) not in after
