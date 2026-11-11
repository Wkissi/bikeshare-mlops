"""Chargement de la configuration et priorité des surcharges."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from bikeshare_mlops.config import CONFIG_FILE_ENV, get_settings


def test_loads_versioned_yaml() -> None:
    settings = get_settings()
    assert settings.project_name == "bikeshare-mlops"
    assert settings.paths.batches_dir == Path("data/batches")
    assert settings.logging.level == "INFO"


def test_env_variable_overrides_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BIKESHARE_LOGGING__LEVEL", "DEBUG")
    assert get_settings().logging.level == "DEBUG"


def test_alternative_config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    custom = tmp_path / "custom.yaml"
    custom.write_text(
        "project_name: autre\n"
        "paths: {data_dir: d, raw_dir: d/r, batches_dir: d/b, incoming_dir: d/i,"
        " reports_dir: r, logs_dir: l}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(CONFIG_FILE_ENV, str(custom))
    assert get_settings().project_name == "autre"


def test_invalid_value_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BIKESHARE_LOGGING__LEVEL", "BAVARD")
    with pytest.raises(ValidationError):
        get_settings()


def test_unknown_key_in_section_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custom = tmp_path / "typo.yaml"
    custom.write_text(
        "project_name: x\n"
        "paths: {data_dir: d, raw_dir: d/r, batches_dir: d/b, incoming_dir: d/i,"
        " reports_dir: r, logs_dir: l, dta_dir: oups}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(CONFIG_FILE_ENV, str(custom))
    with pytest.raises(ValidationError):
        get_settings()
