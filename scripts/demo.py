"""Démonstrations de bout en bout sur la stack Docker démarrée (`make up`).

- nominal  : un lot valide arrive → le worker Prefect entraîne, enregistre et promeut → l'API sert
  le champion → trafic + feedbacks → tout est visible dans Grafana ;
- incident : (a) lot invalide → pipeline bloqué + alerte ; (b) PostgreSQL arrêté → erreurs 503
  propres, /health dégradé, alerte Prometheus ; (c) MLflow arrêté → le modèle reste servi.

Usage : uv run python scripts/demo.py nominal|incident [--hold SECONDES]
"""

from __future__ import annotations

import argparse
import subprocess
import time
from collections.abc import Callable
from typing import TypeVar

import httpx
import pandas as pd
from make_invalid_batch import corrupt
from sqlalchemy.exc import SQLAlchemyError

from bikeshare_mlops.config import Settings, get_settings
from bikeshare_mlops.data.ingest import list_batches, prepare_source, release_next_batches
from bikeshare_mlops.models import registry
from bikeshare_mlops.monitoring.simulate import sample_hours, send_traffic
from bikeshare_mlops.storage import repository
from bikeshare_mlops.storage.db import build_engine, session_factory
from bikeshare_mlops.storage.tables import PipelineRun
from bikeshare_mlops.utils.alerts import alerts_file

API = "http://localhost:8000"
SERVICES = {
    "API": f"{API}/health/live",
    "MLflow": "http://localhost:5000/health",
    "Prefect": "http://localhost:4200/api/health",
    "Prometheus": "http://localhost:9090/-/ready",
    "Grafana": "http://localhost:3000/api/health",
}
BOOTSTRAP_MONTHS = 12
T = TypeVar("T")


def step(title: str) -> None:
    print(f"\n=== {title}")


def wait_for(description: str, probe: Callable[[], T | None], timeout_s: float) -> T:
    """Interroge ``probe`` jusqu'à obtenir une valeur, ou abandonne après ``timeout_s``."""
    deadline = time.monotonic() + timeout_s
    print(f"    … {description} (max {timeout_s:.0f} s)", flush=True)
    while time.monotonic() < deadline:
        value = probe()
        if value is not None:
            return value
        time.sleep(5)
    raise SystemExit(f"Délai dépassé : {description}")


def compose(*args: str) -> None:
    subprocess.run(["docker", "compose", *args], check=True, capture_output=True)


def check_services(client: httpx.Client) -> None:
    step("Services")
    for name, url in SERVICES.items():
        try:
            ok = client.get(url, timeout=5).status_code == 200
        except httpx.HTTPError:
            ok = False
        print(f"    {'ok ' if ok else 'KO '} {name:<11} {url}")
        if not ok:
            raise SystemExit("Stack incomplète : lancer `make up`.")


def latest_run(settings: Settings) -> PipelineRun | None:
    engine = build_engine(settings)
    try:
        with session_factory(engine)() as session:
            return repository.latest_pipeline_run(session)
    except SQLAlchemyError:
        return None
    finally:
        engine.dispose()


def wait_new_run(settings: Settings, previous: PipelineRun | None) -> PipelineRun:
    """Attend qu'un nouveau run de training_pipeline soit enregistré par le worker."""
    previous_id = previous.id if previous else 0

    def probe() -> PipelineRun | None:
        run = latest_run(settings)
        return run if run is not None and run.id > previous_id else None

    return wait_for("le worker Prefect détecte le lot et exécute training_pipeline", probe, 420)


def show_registry(settings: Settings) -> None:
    for row in registry.list_versions(settings)[:5]:
        print(f"    v{row.version:<3} {row.aliases:<20} {row.status:<12} rmse {row.test_rmse}")


def nominal(settings: Settings, client: httpx.Client) -> None:
    check_services(client)

    step("1. Données")
    if not list_batches(settings.paths.batches_dir):
        prepare_source(settings)
    print(f"    {len(list_batches(settings.paths.batches_dir))} lots mensuels disponibles")

    step("2. Arrivée de données")
    registry.configure_tracking(settings)
    has_champion = registry.version_by_alias(settings, registry.CHAMPION) is not None
    count = 1 if has_champion else BOOTSTRAP_MONTHS
    previous = latest_run(settings)
    released = release_next_batches(settings, count)
    if released:
        print(f"    déposé(s) dans incoming : {', '.join(path.name for path in released)}")
        run = wait_new_run(settings, previous)
        print(f"    run {run.state} : {run.outcome} v{run.model_version} ({run.message})")
        if run.state != "COMPLETED":
            raise SystemExit("Le pipeline a échoué : voir `make alerts` et l'UI Prefect.")
    else:
        print("    tous les lots ont déjà été rejoués : on passe au service")

    step("3. Registre MLflow")
    show_registry(settings)
    champion = registry.version_by_alias(settings, registry.CHAMPION)

    step(f"4. L'API sert le champion v{champion}")
    wait_for(
        "chaque worker de l'API suit l'alias champion",
        lambda: True if client.get(f"{API}/model").json().get("version") == champion else None,
        90,
    )
    print(f"    {client.get(f'{API}/health').json()}")

    step("5. Trafic : prédictions puis feedbacks (valeurs réellement observées)")
    source = list_batches(settings.paths.processed_dir)[-1]
    with httpx.Client(base_url=API, timeout=30) as api:
        single = send_traffic(api, sample_hours(source, 300), 0.8, invalid_ratio=0.03)
        batch = send_traffic(api, sample_hours(source, 100, seed=1), 0.5, batch_size=20, seed=1)
    print(f"    unitaires : {single.summary()}")
    print(f"    par lots  : {batch.summary()}")

    step("À regarder")
    print("    Grafana  http://localhost:3000  (dashboard « Bikeshare MLOps », rafraîchi en 5 s)")
    print("    MLflow   http://localhost:5000  (expériences bikeshare-demand, bikeshare-ingestion)")
    print("    Prefect  http://localhost:4200  (runs watch_incoming → training_pipeline)")
    print(f"    API      {API}/docs")


def incident_invalid_batch(settings: Settings) -> None:
    step("(a) Lot invalide : colonnes manquantes, valeurs hors plage, doublons")
    source = list_batches(settings.paths.batches_dir)[0]
    target = settings.paths.incoming_dir / f"batch_invalid_{time.strftime('%Y%m%d%H%M%S')}.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    previous = latest_run(settings)
    corrupt(pd.read_csv(source)).to_csv(target, index=False, lineterminator="\n")
    print(f"    déposé : {target}")
    run = wait_new_run(settings, previous)
    print(f"    run {run.state} : {run.message}")
    print(f"    mis en quarantaine : {settings.paths.rejected_dir / target.name}")
    print(f"    dernière alerte ({alerts_file(settings)}) :")
    print(f"      {alerts_file(settings).read_text(encoding='utf-8').splitlines()[-1][:220]}")


def prometheus_alerts(client: httpx.Client) -> list[str]:
    alerts = client.get("http://localhost:9090/api/v1/alerts").json()["data"]["alerts"]
    return [f"{a['labels']['alertname']} ({a['state']})" for a in alerts]


def incident_database_down(client: httpx.Client, settings: Settings, hold_s: float) -> None:
    step("(b) Dépendance indisponible : arrêt de PostgreSQL")
    compose("stop", "postgres")
    try:
        hours = sample_hours(list_batches(settings.paths.processed_dir)[-1], 30)
        with httpx.Client(base_url=API, timeout=30) as api:
            report = send_traffic(api, hours, feedback_ratio=0)
            print(f"    prédictions pendant la panne : {report.summary()}")
            print(f"    exemple de réponse : {api.post('/predict', json=hours[0][0]).json()}")
            health = api.get("/health")
            print(f"    /health → {health.status_code} {health.json()}")
        wait_for(
            "Prometheus déclenche l'alerte DatabaseDown",
            lambda: (
                True
                if any("DatabaseDown (firing)" in a for a in prometheus_alerts(client))
                else None
            ),
            90,
        )
        print(f"    alertes Prometheus : {prometheus_alerts(client)}")
        print(f"    panne maintenue {hold_s:.0f} s : regarder Grafana (tuile « Base de données »)")
        time.sleep(hold_s)
    finally:
        compose("start", "postgres")
    wait_for(
        "retour à la normale de /health",
        lambda: True if client.get(f"{API}/health").status_code == 200 else None,
        60,
    )
    print("    PostgreSQL redémarré : /health → 200, prédictions de nouveau acceptées")


def incident_registry_down(client: httpx.Client) -> None:
    step("(c) Dépendance indisponible : arrêt de MLflow (registre)")
    compose("stop", "mlflow")
    try:
        reload = client.post(f"{API}/model/reload", timeout=60)
        print(f"    /model/reload → {reload.status_code} {reload.json()['message'][:120]}…")
        payload = sample_hours(list_batches(get_settings().paths.processed_dir)[-1], 1)[0][0]
        predict = client.post(f"{API}/predict", json=payload)
        version = predict.json().get("model_version")
        print(f"    /predict → {predict.status_code} (version {version} toujours servie)")
    finally:
        compose("start", "mlflow")
    print("    MLflow redémarré : le modèle en mémoire n'a jamais cessé d'être servi")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=["nominal", "incident"])
    parser.add_argument("--hold", type=float, default=30, help="durée de la panne PostgreSQL")
    args = parser.parse_args()
    settings = get_settings()
    with httpx.Client(timeout=30) as client:
        if args.scenario == "nominal":
            nominal(settings, client)
        else:
            check_services(client)
            incident_invalid_batch(settings)
            incident_database_down(client, settings, args.hold)
            incident_registry_down(client)
            step("Bilan : http://localhost:3000 (dashboard) et http://localhost:9090/alerts")


if __name__ == "__main__":
    main()
