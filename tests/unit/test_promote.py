"""Règles de promotion : décisions sur des modèles construits à la main (sans MLflow)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from bikeshare_mlops.config import ModelParams, Settings
from bikeshare_mlops.data.dataset import train_test_split_stable
from bikeshare_mlops.features.build import build_pipeline, build_preprocessor, split_features_target
from bikeshare_mlops.models.promote import decide
from factories import make_hourly_frame


class _NegativeRegressor(RegressorMixin, BaseEstimator):
    """Modèle défaillant : prédit des locations négatives."""

    def fit(self, features: pd.DataFrame, target: pd.Series) -> _NegativeRegressor:
        self.fitted_ = True
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return -np.ones(len(features))


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return make_hourly_frame(days=120)


def _fit(model: Pipeline, frame: pd.DataFrame, settings: Settings) -> Pipeline:
    train, _ = train_test_split_stable(
        frame, settings.training.test_fraction, settings.training.split_salt
    )
    return model.fit(*split_features_target(train))


def _forest(n_estimators: int) -> Pipeline:
    return build_pipeline(ModelParams(n_estimators=n_estimators, n_jobs=1, max_depth=None))


def _with(model: BaseEstimator) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])


def test_first_model_is_promoted_when_checks_pass(frame: pd.DataFrame, settings: Settings) -> None:
    challenger = _fit(_forest(20), frame, settings)

    decision = decide("1", challenger, None, None, frame, settings)

    assert decision.promoted
    assert decision.improvement_pct is None
    assert {check.name for check in decision.checks} == {
        "predictions_valid",
        "beats_baseline",
        "single_prediction_latency",
    }


def test_identical_model_is_rejected_for_insufficient_gain(
    frame: pd.DataFrame, settings: Settings
) -> None:
    champion = _fit(_forest(20), frame, settings)
    challenger = _fit(_forest(20), frame, settings)

    decision = decide("2", challenger, champion, "1", frame, settings)

    assert not decision.promoted
    assert decision.improvement_pct == pytest.approx(0.0)
    assert "gain insuffisant" in decision.reason
    assert all(check.passed for check in decision.checks)


def test_clearly_better_model_replaces_champion(frame: pd.DataFrame, settings: Settings) -> None:
    weak_champion = _fit(_with(DummyRegressor()), frame, settings)
    challenger = _fit(_forest(20), frame, settings)

    lenient = settings.model_copy(
        update={
            "promotion": settings.promotion.model_copy(
                update={"max_segment_degradation_pct": 100.0}
            )
        }
    )
    decision = decide("2", challenger, weak_champion, "1", frame, lenient)

    assert decision.promoted
    assert decision.improvement_pct is not None and decision.improvement_pct > 2
    assert decision.as_tags()["promotion_status"] == "promoted"


def test_model_not_beating_baseline_is_rejected(frame: pd.DataFrame, settings: Settings) -> None:
    dummy = _fit(_with(DummyRegressor()), frame, settings)
    decision = decide("1", dummy, None, None, frame, settings)

    assert not decision.promoted
    assert "beats_baseline" in decision.reason


def test_negative_predictions_are_rejected(frame: pd.DataFrame, settings: Settings) -> None:
    decision = decide(
        "1", _fit(_with(_NegativeRegressor()), frame, settings), None, None, frame, settings
    )

    assert not decision.promoted
    assert "predictions_valid" in decision.reason


def test_changed_input_schema_is_rejected(frame: pd.DataFrame, settings: Settings) -> None:
    champion = _fit(_forest(10), frame, settings)
    train, _ = train_test_split_stable(frame, 0.2, settings.training.split_salt)
    features, target = split_features_target(train)
    reordered = _forest(10).fit(features[features.columns[::-1]], target)

    decision = decide("2", reordered, champion, "1", frame, settings)

    assert not decision.promoted
    assert "same_input_schema" in decision.reason


def test_too_slow_model_is_rejected(frame: pd.DataFrame, settings: Settings) -> None:
    strict = settings.model_copy(
        update={
            "promotion": settings.promotion.model_copy(update={"max_single_prediction_ms": 1e-6})
        }
    )
    decision = decide("1", _fit(_forest(10), frame, settings), None, None, frame, strict)

    assert not decision.promoted
    assert "single_prediction_latency" in decision.reason
