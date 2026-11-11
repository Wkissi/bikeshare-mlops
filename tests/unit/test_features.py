"""Prétraitement : encodage cyclique, absence de fuite, pipeline sérialisable."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from bikeshare_mlops.config import ModelParams
from bikeshare_mlops.features.build import (
    FEATURES,
    CyclicalEncoder,
    build_pipeline,
    build_preprocessor,
    split_features_target,
)

FAST = ModelParams(n_estimators=5, n_jobs=1)


def test_cyclical_encoding_makes_midnight_and_23h_neighbours() -> None:
    encoder = CyclicalEncoder(period=24).fit(pd.DataFrame({"hr": [0, 23, 12]}))
    encoded = encoder.transform(pd.DataFrame({"hr": [0, 23, 12]}))

    def distance(a: int, b: int) -> float:
        return float(np.linalg.norm(encoded[a] - encoded[b]))

    assert distance(0, 1) < distance(0, 2)
    assert list(encoder.get_feature_names_out()) == ["hr_sin", "hr_cos"]


def test_target_leaking_columns_are_not_features() -> None:
    assert {"cnt", "casual", "registered", "instant", "dteday"}.isdisjoint(FEATURES)


def test_preprocessor_output_is_numeric_and_named(hourly_frame: pd.DataFrame) -> None:
    features, _ = split_features_target(hourly_frame)
    preprocessor = build_preprocessor().fit(features)
    output = preprocessor.transform(features)

    names = list(preprocessor.get_feature_names_out())
    assert output.shape == (len(features), len(names))
    assert np.isfinite(output).all()
    assert {"hr_sin", "mnth_cos", "weathersit_1", "temp"}.issubset(names)


def test_unknown_category_does_not_break_inference(hourly_frame: pd.DataFrame) -> None:
    features, target = split_features_target(hourly_frame)
    pipeline = build_pipeline(FAST).fit(features, target)

    unseen = features.head(1).assign(weathersit=4)
    assert pipeline.predict(unseen).shape == (1,)


def test_pipeline_roundtrip_gives_identical_predictions(hourly_frame: pd.DataFrame) -> None:
    features, target = split_features_target(hourly_frame)
    pipeline = build_pipeline(FAST).fit(features, target)

    restored = pickle.loads(pickle.dumps(pipeline))
    np.testing.assert_allclose(restored.predict(features), pipeline.predict(features))


def test_missing_feature_column_fails_loudly(hourly_frame: pd.DataFrame) -> None:
    with pytest.raises(KeyError):
        split_features_target(hourly_frame.drop(columns=["temp"]))
