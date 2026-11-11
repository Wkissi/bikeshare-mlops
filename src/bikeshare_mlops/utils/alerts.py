"""Alertes : toujours tracées dans un fichier JSONL, envoyées à un webhook s'il est configuré.

Un webhook en panne ne doit jamais masquer l'incident d'origine : l'échec d'envoi est journalisé,
pas relevé.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from bikeshare_mlops.config import Settings, get_settings
from bikeshare_mlops.utils.logging import get_logger

logger = get_logger(__name__)


def alerts_file(settings: Settings) -> Path:
    return settings.paths.logs_dir / settings.alerts.file_name


def send_alert(
    settings: Settings,
    event: str,
    message: str,
    context: dict[str, Any] | None = None,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Enregistre l'alerte et la pousse au webhook ; renvoie l'enregistrement produit."""
    record = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "event": event,
        "message": message,
        **(context or {}),
    }
    path = alerts_file(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    logger.error("alerte", extra={"alert": record})

    url = settings.alerts.webhook_url
    if url:
        text = f"[{settings.project_name}] {event} : {message}"
        # Slack lit « text », Discord lit « content » : les deux sont fournis.
        payload = {"text": text, "content": text, "alert": record}
        http = client or httpx.Client(timeout=settings.alerts.timeout_s)
        try:
            http.post(url, json=payload).raise_for_status()
        except httpx.HTTPError:
            logger.exception("envoi de l'alerte au webhook impossible", extra={"event": event})
        finally:
            if client is None:
                http.close()
    return record


def main(argv: list[str] | None = None) -> None:
    """Affiche les dernières alertes enregistrées."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--tail", type=int, default=5)
    args = parser.parse_args(argv)
    path = alerts_file(get_settings())
    if not path.exists():
        print("Aucune alerte.")
        return
    for line in path.read_text(encoding="utf-8").splitlines()[-args.tail :]:
        record = json.loads(line)
        print(f"{record['ts']}  {record['event']}  {record['message']}")


if __name__ == "__main__":
    main()
