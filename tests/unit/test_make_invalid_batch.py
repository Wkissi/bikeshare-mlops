"""Le lot de démonstration « invalide » est bien bloqué par les contrôles qualité."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from bikeshare_mlops.config import Settings
from bikeshare_mlops.data.validate import validate_frame
from bikeshare_mlops.exceptions import DataValidationError
from factories import corrupt_frame


def test_invalid_batch_triggers_every_family_of_checks(
    hourly_frame: pd.DataFrame, settings: Settings
) -> None:
    with pytest.raises(DataValidationError) as caught:
        validate_frame(corrupt_frame(hourly_frame), "batch_invalid", settings)

    checks = json.loads(caught.value.report_path.read_text(encoding="utf-8"))["failures_by_check"]
    expected_checks = ("column_in_dataframe", "in_range", "not_nullable", "fields_uniqueness")
    for expected in expected_checks:
        assert any(expected in key for key in checks), checks
