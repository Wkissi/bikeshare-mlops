"""Simulateur de trafic contre l'API réelle (TestClient, modèle factice, SQLite)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine

from bikeshare_mlops.api.main import create_app
from bikeshare_mlops.api.model_service import ModelService
from bikeshare_mlops.config import get_settings
from bikeshare_mlops.monitoring.simulate import main, sample_hours, send_traffic
from factories import make_hourly_frame

pytestmark = pytest.mark.integration


class _HourModel:
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame["hr"].to_numpy(dtype=float) * 10


@pytest.fixture
def client(db_engine: Engine) -> Iterator[TestClient]:
    service = ModelService("bikeshare-rf", "champion", lambda: (_HourModel(), "1"))
    with TestClient(create_app(get_settings(), service, db_engine)) as test_client:
        yield test_client


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "batch_2011_01.csv"
    make_hourly_frame(days=3).to_csv(path, index=False)
    return path


def test_feedbacks_give_the_true_mae(client: TestClient, source: Path) -> None:
    hours = sample_hours(source, 40)
    report = send_traffic(client, hours, feedback_ratio=1.0)

    expected = np.mean([abs(hour["hr"] * 10 - actual) for hour, actual in hours])
    assert report.status_codes == {200: 40}
    assert report.feedbacks == 40
    assert report.mae == pytest.approx(expected, abs=1e-2)
    assert "feedbacks : 40" in report.summary()


def test_invalid_requests_are_mixed_in(client: TestClient, source: Path) -> None:
    report = send_traffic(client, sample_hours(source, 40), invalid_ratio=0.25, seed=1)

    assert report.status_codes[422] > 0
    assert report.status_codes[200] + report.status_codes[422] == 40


def test_batch_mode(client: TestClient, source: Path) -> None:
    report = send_traffic(client, sample_hours(source, 30), feedback_ratio=0.0, batch_size=10)
    assert report.status_codes == {200: 3}
    assert report.feedbacks == 0
    assert report.mae is None


def test_cli_requires_data(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit, match="make data"):
        main(["--count", "1"])
