"""Hooks d'état Prefect partagés par les flows : alerte et traçage des runs en base."""

from __future__ import annotations

from prefect import Flow
from prefect.client.schemas.objects import FlowRun
from prefect.states import State
from sqlalchemy.exc import SQLAlchemyError

from bikeshare_mlops.config import get_settings
from bikeshare_mlops.storage import repository
from bikeshare_mlops.storage.db import build_engine, session_factory
from bikeshare_mlops.utils.alerts import send_alert
from bikeshare_mlops.utils.logging import get_logger

logger = get_logger(__name__)


def record_run(
    flow_name: str,
    flow_run_id: str,
    state: str,
    outcome: str | None = None,
    model_version: str | None = None,
    message: str | None = None,
) -> None:
    """Trace l'issue d'un run dans ``pipeline_runs`` ; une base injoignable ne fait pas échouer
    le flow (l'information reste dans Prefect et dans les logs)."""
    engine = build_engine(get_settings())
    try:
        with session_factory(engine)() as session:
            repository.record_pipeline_run(
                session, flow_name, flow_run_id, state, outcome, model_version, message
            )
    except SQLAlchemyError:
        logger.warning("run de pipeline non enregistré : base indisponible", extra={"state": state})
    finally:
        engine.dispose()


def alert_on_failure(flow: Flow, flow_run: FlowRun, state: State) -> None:
    """Alerte et trace l'échec (lot invalide, MLflow injoignable après retries, etc.)."""
    message = state.message or "échec du flow"
    send_alert(
        get_settings(),
        event="pipeline_failed",
        message=message,
        context={
            "flow": flow.name,
            "flow_run": flow_run.name,
            "flow_run_id": str(flow_run.id),
            "parameters": flow_run.parameters,
        },
    )
    record_run(flow.name, str(flow_run.id), "FAILED", message=message[:2000])
