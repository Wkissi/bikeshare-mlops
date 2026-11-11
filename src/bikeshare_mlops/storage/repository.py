"""Accès aux données : seules ces fonctions écrivent ou lisent les tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from bikeshare_mlops.storage.tables import Feedback, PipelineRun, Prediction


@dataclass(frozen=True)
class PredictionRecord:
    """Prédiction à enregistrer (succès : ``prediction`` renseignée ; échec : ``error``)."""

    id: str
    inputs: dict[str, Any]
    prediction: float | None
    model_name: str | None
    model_version: str | None
    latency_ms: float
    status: str
    error: str | None = None
    batch_id: str | None = None


def save_predictions(session: Session, records: list[PredictionRecord]) -> None:
    session.add_all(
        Prediction(
            id=record.id,
            batch_id=record.batch_id,
            inputs=record.inputs,
            prediction=record.prediction,
            model_name=record.model_name,
            model_version=record.model_version,
            latency_ms=record.latency_ms,
            status=record.status,
            error=record.error,
        )
        for record in records
    )
    session.commit()


def get_prediction(session: Session, prediction_id: str) -> Prediction | None:
    return session.get(Prediction, prediction_id)


def get_feedback(session: Session, prediction_id: str) -> Feedback | None:
    return session.scalar(select(Feedback).where(Feedback.prediction_id == prediction_id))


def save_feedback(session: Session, prediction: Prediction, actual: float) -> Feedback:
    """Enregistre la valeur observée et l'erreur absolue de la prédiction correspondante."""
    feedback = Feedback(
        prediction_id=prediction.id,
        actual=actual,
        absolute_error=abs(actual - (prediction.prediction or 0.0)),
    )
    session.add(feedback)
    session.commit()
    return feedback


def rolling_mae(session: Session, window: int) -> tuple[float | None, int]:
    """MAE sur les ``window`` derniers feedbacks ; renvoie (mae, nombre de feedbacks utilisés)."""
    latest = (
        select(Feedback.absolute_error)
        .order_by(Feedback.created_at.desc(), Feedback.id.desc())
        .limit(window)
        .subquery()
    )
    mae, count = session.execute(
        select(func.avg(latest.c.absolute_error), func.count(latest.c.absolute_error))
    ).one()
    return (None if mae is None else float(mae)), int(count)


def record_pipeline_run(
    session: Session,
    flow_name: str,
    flow_run_id: str,
    state: str,
    outcome: str | None = None,
    model_version: str | None = None,
    message: str | None = None,
) -> None:
    session.add(
        PipelineRun(
            flow_name=flow_name,
            flow_run_id=flow_run_id,
            state=state,
            outcome=outcome,
            model_version=model_version,
            message=message,
        )
    )
    session.commit()


def pipeline_run_counts(session: Session) -> dict[str, int]:
    """Nombre de runs par état (COMPLETED, FAILED)."""
    rows = session.execute(select(PipelineRun.state, func.count()).group_by(PipelineRun.state))
    return {state: int(count) for state, count in rows}


def latest_pipeline_run(session: Session) -> PipelineRun | None:
    return session.scalar(
        select(PipelineRun).order_by(PipelineRun.created_at.desc(), PipelineRun.id.desc()).limit(1)
    )
