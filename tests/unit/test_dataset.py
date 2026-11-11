"""Assemblage des lots et découpage train/test stable."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bikeshare_mlops.data.dataset import is_test_row, load_batches, train_test_split_stable
from factories import make_hourly_frame


def test_split_is_deterministic_and_close_to_fraction() -> None:
    frame = make_hourly_frame(days=90)
    first = is_test_row(frame, 0.2, "sel")
    second = is_test_row(frame, 0.2, "sel")

    assert first.equals(second)
    assert 0.17 < first.mean() < 0.23


def test_split_is_stable_when_new_data_arrives() -> None:
    january = make_hourly_frame(days=31)
    january_and_february = make_hourly_frame(days=59)

    _, test_before = train_test_split_stable(january, 0.2, "sel")
    _, test_after = train_test_split_stable(january_and_february, 0.2, "sel")

    keys_before = set(zip(test_before["dteday"], test_before["hr"], strict=True))
    keys_after = set(zip(test_after["dteday"], test_after["hr"], strict=True))
    assert keys_before <= keys_after


def test_train_and_test_are_disjoint() -> None:
    train, test = train_test_split_stable(make_hourly_frame(days=31), 0.2, "sel")
    merged = train.merge(test, on=["dteday", "hr"])
    assert merged.empty


def test_load_batches_concatenates_in_order(tmp_path: Path) -> None:
    frame = make_hourly_frame(days=59)
    frame.iloc[744:].to_csv(tmp_path / "batch_2011_02.csv", index=False)
    frame.iloc[:744].to_csv(tmp_path / "batch_2011_01.csv", index=False)

    loaded = load_batches(tmp_path)
    assert len(loaded) == len(frame)
    pd.testing.assert_series_equal(loaded["dteday"], frame["dteday"], check_dtype=False)


def test_load_batches_requires_data(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_batches(tmp_path)
