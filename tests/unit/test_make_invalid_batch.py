"""Le lot de démonstration « invalide » est bien bloqué par les contrôles qualité."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from bikeshare_mlops.config import Settings
from bikeshare_mlops.data.validate import validate_frame
from bikeshare_mlops.exceptions import DataValidationError

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "make_invalid_batch.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("make_invalid_batch", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_invalid_batch_triggers_every_family_of_checks(
    hourly_frame: pd.DataFrame, settings: Settings
) -> None:
    corrupted = _load_script().corrupt(hourly_frame)

    with pytest.raises(DataValidationError) as caught:
        validate_frame(corrupted, "batch_invalid", settings)

    checks = json.loads(caught.value.report_path.read_text(encoding="utf-8"))["failures_by_check"]
    expected_checks = ("column_in_dataframe", "in_range", "not_nullable", "fields_uniqueness")
    for expected in expected_checks:
        assert any(expected in key for key in checks), checks
