"""Format JSON des logs structurés."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator

import pytest

from bikeshare_mlops.utils.logging import configure_logging, get_logger


@pytest.fixture(autouse=True)
def _restore_root_logger() -> Iterator[None]:
    """Remet le logger racine en état : les handlers testés écrivent dans un flux temporaire."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers, root.level = handlers, level


def test_json_log_contains_extra_fields(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", json_output=True)
    get_logger("test").info("lot ingéré", extra={"batch": "2011_01", "rows": 688})

    line = capsys.readouterr().err.strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload["event"] == "lot ingéré"
    assert payload["level"] == "INFO"
    assert payload["batch"] == "2011_01"
    assert payload["rows"] == 688


def test_exception_is_serialized(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", json_output=True)
    try:
        raise ValueError("boom")
    except ValueError:
        get_logger("test").exception("échec")

    payload = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert "ValueError: boom" in payload["exception"]


def test_level_filters_messages(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("WARNING", json_output=False)
    logger = get_logger("test")
    logger.info("invisible")
    logger.warning("visible")

    err = capsys.readouterr().err
    assert "invisible" not in err
    assert "visible" in err
