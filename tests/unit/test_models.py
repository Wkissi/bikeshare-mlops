"""Baseline, métriques et entraînement de bout en bout sur données synthétiques."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bikeshare_mlops.config import ModelParams, Settings
from bikeshare_mlops.exceptions import InsufficientDataError
from bikeshare_mlops.features.build import split_features_target
from bikeshare_mlops.models import train
from bikeshare_mlops.models.baseline import HourlyMeanBaseline
from bikeshare_mlops.models.evaluate import compute_metrics, metrics_by_segment
from bikeshare_mlops.models.train import train_and_evaluate
from factories import make_hourly_frame


@pytest.fixture
def fast_settings(settings: Settings) -> Settings:
    """Forêt réduite pour garder les tests rapides."""
    fast = ModelParams(n_estimators=10, n_jobs=1)
    training = settings.training.model_copy(update={"model": fast})
    return settings.model_copy(update={"training": training})


def test_metrics_on_known_values() -> None:
    metrics = compute_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0]))
    assert metrics.mae == pytest.approx(2 / 3)
    assert metrics.rmse == pytest.approx(np.sqrt(4 / 3))
    assert metrics.as_dict("test_").keys() == {"test_rmse", "test_mae", "test_r2"}


def test_baseline_predicts_group_means_with_fallback() -> None:
    features = pd.DataFrame({"hr": [8, 8, 9], "workingday": [1, 1, 0]})
    baseline = HourlyMeanBaseline().fit(features, pd.Series([10.0, 20.0, 6.0]))

    unseen = pd.DataFrame({"hr": [8, 3], "workingday": [1, 1]})
    np.testing.assert_allclose(baseline.predict(unseen), [15.0, 12.0])


def test_metrics_by_segment_cover_each_season() -> None:
    frame = make_hourly_frame(days=120)
    segments = metrics_by_segment(frame, frame["cnt"].to_numpy(dtype=float), "cnt")
    assert set(segments) == {str(s) for s in frame["season"].unique()}
    assert all(m.mae == 0 for m in segments.values())


def test_training_beats_baseline_and_writes_artifacts(
    fast_settings: Settings, tmp_path: Path
) -> None:
    frame = make_hourly_frame(days=120)
    result = train_and_evaluate(frame, fast_settings, tmp_path / "run")

    assert result.train_rows + result.test_rows == len(frame)
    assert result.metrics.rmse < result.baseline_metrics.rmse
    names = {path.name for path in result.artifacts}
    assert {"metrics.json", "predicted_vs_actual.png", "feature_importances.png"} <= names
    summary = json.loads((tmp_path / "run" / "metrics.json").read_text(encoding="utf-8"))
    assert summary["data_sha256"] == result.data_sha256


def test_trained_pipeline_predicts_raw_rows(fast_settings: Settings, tmp_path: Path) -> None:
    result = train_and_evaluate(make_hourly_frame(days=60), fast_settings, tmp_path)
    features, _ = split_features_target(result.test_frame.head(3))
    assert (result.pipeline.predict(features) >= 0).all()


def test_too_little_data_is_refused(fast_settings: Settings, tmp_path: Path) -> None:
    with pytest.raises(InsufficientDataError):
        train_and_evaluate(make_hourly_frame(days=2), fast_settings, tmp_path)


def test_cli_trains_from_batch_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    batches = tmp_path / "batches"
    batches.mkdir()
    make_hourly_frame(days=60).to_csv(batches / "batch_2011_01.csv", index=False)
    monkeypatch.setenv("BIKESHARE_TRAINING__MODEL__N_ESTIMATORS", "5")
    monkeypatch.setenv("BIKESHARE_TRAINING__MODEL__N_JOBS", "1")

    train.main(["--no-mlflow", "--data-dir", str(batches), "--output-dir", str(tmp_path / "out")])

    assert "test_rmse" in json.loads(capsys.readouterr().out)
    assert (tmp_path / "out" / "residuals.png").exists()
