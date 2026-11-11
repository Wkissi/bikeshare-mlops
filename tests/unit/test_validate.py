"""Contrôles qualité : un lot conforme passe, chaque type d'anomalie bloque avec un rapport."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bikeshare_mlops.config import Settings
from bikeshare_mlops.data.validate import main, validate_file, validate_frame
from bikeshare_mlops.exceptions import DataValidationError


def _report(error: DataValidationError) -> dict:
    return json.loads(error.report_path.read_text(encoding="utf-8"))


def test_valid_batch_passes_and_writes_report(
    hourly_frame: pd.DataFrame, settings: Settings
) -> None:
    result = validate_frame(hourly_frame, "batch_2011_01", settings)

    assert len(result.data) == len(hourly_frame)
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["n_failures"] == 0


def _drop_column(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=["hum"])


def _hour_out_of_range(frame: pd.DataFrame) -> pd.DataFrame:
    frame.loc[0, "hr"] = 24
    return frame


def _temp_out_of_range(frame: pd.DataFrame) -> pd.DataFrame:
    frame.loc[0, "temp"] = 1.5
    return frame


def _null_value(frame: pd.DataFrame) -> pd.DataFrame:
    frame["windspeed"] = frame["windspeed"].astype("float64")
    frame.loc[0, "windspeed"] = np.nan
    return frame


def _duplicate_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([frame, frame.head(2)], ignore_index=True)


def _inconsistent_counts(frame: pd.DataFrame) -> pd.DataFrame:
    frame.loc[0, "cnt"] = frame.loc[0, "cnt"] + 1
    return frame


def _too_few_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.head(10)


def _bad_date(frame: pd.DataFrame) -> pd.DataFrame:
    frame.loc[0, "dteday"] = "01/01/2011"
    return frame


def _unexpected_column(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.assign(extra=1)


@pytest.mark.parametrize(
    ("corruption", "expected_check"),
    [
        (_drop_column, "column_in_dataframe"),
        (_hour_out_of_range, "in_range"),
        (_temp_out_of_range, "in_range"),
        (_null_value, "not_nullable"),
        (_duplicate_rows, "multiple_fields_uniqueness"),
        (_inconsistent_counts, "cnt_equals_casual_plus_registered"),
        (_too_few_rows, "min_rows"),
        (_bad_date, "iso_date"),
        (_unexpected_column, "column_in_schema"),
    ],
)
def test_each_anomaly_blocks_with_report(
    hourly_frame: pd.DataFrame,
    settings: Settings,
    corruption: Callable[[pd.DataFrame], pd.DataFrame],
    expected_check: str,
) -> None:
    with pytest.raises(DataValidationError) as caught:
        validate_frame(corruption(hourly_frame.copy()), "batch_bad", settings)

    report = _report(caught.value)
    assert report["status"] == "failed"
    assert caught.value.n_failures >= 1
    assert any(expected_check in key for key in report["failures_by_check"]), report


def test_unreadable_file_is_rejected(settings: Settings, tmp_path: Path) -> None:
    empty = tmp_path / "batch_empty.csv"
    empty.write_text("", encoding="utf-8")

    with pytest.raises(DataValidationError) as caught:
        validate_file(empty, settings)
    assert _report(caught.value)["failures_by_check"] == {"file:readable_csv": 1}


def test_validate_file_reads_csv(
    hourly_frame: pd.DataFrame, settings: Settings, tmp_path: Path
) -> None:
    path = tmp_path / "batch_2011_01.csv"
    hourly_frame.to_csv(path, index=False)

    assert validate_file(path, settings).batch_name == "batch_2011_01"


def test_cli_exit_code_reflects_conformity(
    hourly_frame: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    good, bad = tmp_path / "batch_ok.csv", tmp_path / "batch_ko.csv"
    hourly_frame.to_csv(good, index=False)
    hourly_frame.drop(columns=["cnt"]).to_csv(bad, index=False)

    main([str(good)])
    with pytest.raises(SystemExit) as caught:
        main([str(good), str(bad)])
    assert caught.value.code == 1
