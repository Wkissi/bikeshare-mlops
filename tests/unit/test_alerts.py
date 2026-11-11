"""Alertes : fichier JSONL toujours écrit, webhook optionnel et jamais bloquant."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from bikeshare_mlops.config import AlertsConfig, Settings, get_settings
from bikeshare_mlops.utils.alerts import alerts_file, main, send_alert


def _with_webhook(settings: Settings, url: str | None) -> Settings:
    return settings.model_copy(update={"alerts": AlertsConfig(webhook_url=url)})


def test_alert_is_written_to_file_without_webhook(settings: Settings) -> None:
    send_alert(_with_webhook(settings, None), "pipeline_failed", "lot invalide", {"batch": "b1"})

    lines = alerts_file(settings).read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[-1])
    assert record["event"] == "pipeline_failed"
    assert record["batch"] == "b1"


def test_alert_is_posted_to_webhook(settings: Settings) -> None:
    received: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        received.append(json.loads(request.content))
        return httpx.Response(204)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    send_alert(_with_webhook(settings, "https://hooks.test/x"), "e", "m", client=client)

    assert received[0]["text"] == received[0]["content"]
    assert "e : m" in received[0]["text"]


def test_cli_shows_last_alerts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("BIKESHARE_PATHS__LOGS_DIR", str(tmp_path / "logs"))
    main([])
    assert "Aucune alerte." in capsys.readouterr().out

    get_settings.cache_clear()
    send_alert(get_settings(), "pipeline_failed", "lot invalide")
    main(["--tail", "1"])
    assert "pipeline_failed  lot invalide" in capsys.readouterr().out


def test_failing_webhook_does_not_raise(settings: Settings) -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(500)))

    record = send_alert(_with_webhook(settings, "https://hooks.test/x"), "e", "m", client=client)

    assert record["event"] == "e"
    assert alerts_file(settings).exists()
