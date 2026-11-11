"""Métriques Prometheus de l'API.

Trois familles :
- trafic HTTP (prometheus-fastapi-instrumentator) : requêtes par route et code, histogramme de
  latence (p50/p95/p99 calculés dans Grafana) ;
- métier, incrémentées par l'API : prédictions par version et statut, feedbacks, erreurs par type,
  temps d'inférence du modèle ;
- état, recalculé depuis PostgreSQL à chaque collecte : MAE glissante des feedbacks, dernier run
  de pipeline, disponibilité de la base.

Avec plusieurs workers uvicorn, ``PROMETHEUS_MULTIPROC_DIR`` active le mode multiprocessus :
chaque worker écrit ses compteurs dans ce dossier et ``/metrics`` agrège tous les workers.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

from fastapi import FastAPI, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    multiprocess,
)
from prometheus_client.core import GaugeMetricFamily, Metric
from prometheus_client.registry import Collector
from prometheus_fastapi_instrumentator import Instrumentator
from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from bikeshare_mlops.storage import repository
from bikeshare_mlops.storage.db import is_reachable

MULTIPROC_ENV = "PROMETHEUS_MULTIPROC_DIR"
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1, 2.5, 5)

REGISTRY = CollectorRegistry(auto_describe=True)

PREDICTIONS = Counter(
    "bikeshare_predictions",
    "Prédictions servies, par version du modèle, statut et route",
    ["model_version", "status", "endpoint"],
    registry=REGISTRY,
)
FEEDBACKS = Counter("bikeshare_feedbacks", "Feedbacks enregistrés", registry=REGISTRY)
API_ERRORS = Counter(
    "bikeshare_api_errors", "Erreurs renvoyées par l'API, par type", ["error"], registry=REGISTRY
)
INFERENCE_SECONDS = Histogram(
    "bikeshare_model_inference_seconds",
    "Durée de l'inférence seule (hors HTTP et base)",
    buckets=LATENCY_BUCKETS,
    registry=REGISTRY,
)
SERVED_VERSION = Gauge(
    "bikeshare_served_model_version",
    "Version du modèle servie (0 = aucun modèle chargé)",
    registry=REGISTRY,
    multiprocess_mode="livemax",
)


def record_served_version(version: str) -> None:
    """Met à jour la jauge de version servie (numéro de version du registre)."""
    SERVED_VERSION.set(int(version) if version.isdigit() else 0)


class DatabaseCollector(Collector):
    """Indicateurs calculés depuis la base au moment de la collecte (cohérents entre workers)."""

    def __init__(self, engine: Engine, sessions: sessionmaker[Session], mae_window: int) -> None:
        self.engine = engine
        self.sessions = sessions
        self.mae_window = mae_window

    def collect(self) -> Iterator[Metric]:
        database_up = is_reachable(self.engine)
        yield GaugeMetricFamily(
            "bikeshare_database_up", "1 si PostgreSQL répond", value=float(database_up)
        )
        if not database_up:
            return
        try:
            with self.sessions() as session:
                mae, samples = repository.rolling_mae(session, self.mae_window)
                last_run = repository.latest_pipeline_run(session)
                run_counts = repository.pipeline_run_counts(session)
        except SQLAlchemyError:
            return

        yield GaugeMetricFamily(
            "bikeshare_rolling_mae",
            f"MAE sur les {self.mae_window} derniers feedbacks (NaN sans feedback)",
            value=float("nan") if mae is None else mae,
        )
        yield GaugeMetricFamily(
            "bikeshare_rolling_mae_samples", "Feedbacks utilisés pour la MAE", value=samples
        )
        runs = GaugeMetricFamily(
            "bikeshare_pipeline_runs", "Runs de training_pipeline par état", labels=["state"]
        )
        for state, count in run_counts.items():
            runs.add_metric([state], count)
        yield runs
        if last_run is not None:
            yield GaugeMetricFamily(
                "bikeshare_last_pipeline_run_success",
                "1 si le dernier run de training_pipeline a réussi",
                value=float(last_run.state == "COMPLETED"),
            )
            yield GaugeMetricFamily(
                "bikeshare_last_pipeline_run_timestamp_seconds",
                "Date du dernier run de training_pipeline",
                value=last_run.created_at.timestamp(),
            )


def _counters_payload(http_registry: CollectorRegistry) -> bytes:
    """Compteurs et histogrammes : agrégés sur tous les workers en mode multiprocessus."""
    if MULTIPROC_ENV in os.environ:
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return generate_latest(registry)
    return generate_latest(REGISTRY) + generate_latest(http_registry)


def install_metrics(app: FastAPI, mae_window: int) -> None:
    """Instrumente les routes et expose ``GET /metrics``.

    Les métriques HTTP ont un registre propre à l'application (plusieurs applications peuvent
    coexister dans un même processus, ex. les tests).
    """
    http_registry = CollectorRegistry(auto_describe=True)
    Instrumentator(
        should_group_status_codes=False,
        excluded_handlers=["/metrics", "/health/live"],
        registry=http_registry,
    ).instrument(app, latency_lowr_buckets=LATENCY_BUCKETS)

    @app.get("/metrics", include_in_schema=False)
    def metrics(request: Request) -> Response:
        state_registry = CollectorRegistry()
        state_registry.register(
            DatabaseCollector(request.app.state.engine, request.app.state.sessions, mae_window)
        )
        payload = _counters_payload(http_registry) + generate_latest(state_registry)
        return Response(payload, media_type=CONTENT_TYPE_LATEST)
