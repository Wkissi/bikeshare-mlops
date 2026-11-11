"""Hooks d'état Prefect partagés par les flows."""

from __future__ import annotations

from prefect import Flow
from prefect.client.schemas.objects import FlowRun
from prefect.states import State

from bikeshare_mlops.config import get_settings
from bikeshare_mlops.utils.alerts import send_alert


def alert_on_failure(flow: Flow, flow_run: FlowRun, state: State) -> None:
    """Alerte dès qu'un flow échoue (lot invalide, MLflow injoignable après retries, etc.)."""
    send_alert(
        get_settings(),
        event="pipeline_failed",
        message=state.message or "échec du flow",
        context={
            "flow": flow.name,
            "flow_run": flow_run.name,
            "flow_run_id": str(flow_run.id),
            "parameters": flow_run.parameters,
        },
    )
