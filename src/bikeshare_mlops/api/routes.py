"""Routes de l'API de prévision."""

from __future__ import annotations

import time
from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from bikeshare_mlops.api.errors import ApiError
from bikeshare_mlops.api.model_service import ModelService
from bikeshare_mlops.api.schemas import (
    BatchItem,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ComponentStatus,
    ErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    ModelInfo,
    PredictionResponse,
    RentalFeatures,
)
from bikeshare_mlops.monitoring.metrics import FEEDBACKS, INFERENCE_SECONDS, PREDICTIONS
from bikeshare_mlops.storage import repository
from bikeshare_mlops.storage.db import is_reachable
from bikeshare_mlops.storage.repository import PredictionRecord
from bikeshare_mlops.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

_ERRORS: dict[int | str, dict[str, Any]] = {
    422: {"model": ErrorResponse, "description": "Entrée invalide"},
    503: {"model": ErrorResponse, "description": "Modèle ou base de données indisponible"},
    500: {"model": ErrorResponse, "description": "Erreur interne"},
}


def _service(request: Request) -> ModelService:
    return request.app.state.model_service


def _sessions(request: Request) -> sessionmaker[Session]:
    return request.app.state.sessions


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)


def _predict_and_store(
    request: Request, rows: list[dict[str, Any]], batch_id: str | None
) -> tuple[list[PredictionRecord], str]:
    """Prédit puis enregistre ; un échec du modèle est lui aussi tracé (si la base répond)."""
    service = _service(request)
    endpoint = "batch" if batch_id else "single"
    start = time.perf_counter()
    ids = [str(uuid4()) for _ in rows]
    try:
        with INFERENCE_SECONDS.time():
            values, version = service.predict(rows)
    except Exception as error:
        current = service.current
        PREDICTIONS.labels(
            model_version=current.version if current else "none",
            status="error",
            endpoint=endpoint,
        ).inc(len(rows))
        failures = [
            PredictionRecord(
                id=prediction_id,
                inputs=row,
                prediction=None,
                model_name=service.model_name,
                model_version=current.version if current else None,
                latency_ms=_elapsed_ms(start),
                status="error",
                error=f"{type(error).__name__}: {error}",
                batch_id=batch_id,
            )
            for prediction_id, row in zip(ids, rows, strict=True)
        ]
        try:
            with _sessions(request)() as session:
                repository.save_predictions(session, failures)
        except SQLAlchemyError:
            logger.warning("échec de prédiction non enregistré : base indisponible")
        raise

    latency = _elapsed_ms(start)
    records = [
        PredictionRecord(
            id=prediction_id,
            inputs=row,
            prediction=round(float(value), 3),
            model_name=service.model_name,
            model_version=version,
            latency_ms=latency,
            status="success",
            batch_id=batch_id,
        )
        for prediction_id, row, value in zip(ids, rows, values, strict=True)
    ]
    with _sessions(request)() as session:
        repository.save_predictions(session, records)
    PREDICTIONS.labels(model_version=version, status="success", endpoint=endpoint).inc(len(rows))
    return records, version


@router.post("/predict", response_model=PredictionResponse, responses=_ERRORS, tags=["prédiction"])
def predict(payload: RentalFeatures, request: Request) -> PredictionResponse:
    """Prévision du nombre de locations pour une heure."""
    records, version = _predict_and_store(request, [payload.model_dump()], batch_id=None)
    record = records[0]
    return PredictionResponse(
        prediction_id=UUID(record.id),
        prediction=record.prediction or 0.0,
        model_name=_service(request).model_name,
        model_version=version,
        latency_ms=record.latency_ms,
    )


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    responses=_ERRORS,
    tags=["prédiction"],
)
def predict_batch(payload: BatchPredictionRequest, request: Request) -> BatchPredictionResponse:
    """Prévisions pour plusieurs heures en un appel (une ligne tracée par heure)."""
    batch_id = str(uuid4())
    rows = [instance.model_dump() for instance in payload.instances]
    records, version = _predict_and_store(request, rows, batch_id=batch_id)
    return BatchPredictionResponse(
        batch_id=UUID(batch_id),
        model_name=_service(request).model_name,
        model_version=version,
        latency_ms=records[0].latency_ms,
        predictions=[
            BatchItem(prediction_id=UUID(record.id), prediction=record.prediction or 0.0)
            for record in records
        ],
    )


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    responses={**_ERRORS, 404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    tags=["feedback"],
)
def feedback(payload: FeedbackRequest, request: Request) -> FeedbackResponse:
    """Enregistre la valeur réellement observée pour une prédiction passée."""
    prediction_id = str(payload.prediction_id)
    with _sessions(request)() as session:
        prediction = repository.get_prediction(session, prediction_id)
        if prediction is None:
            raise ApiError(404, "prediction_not_found", f"Prédiction {prediction_id} inconnue.")
        if prediction.status != "success" or prediction.prediction is None:
            raise ApiError(
                409, "prediction_failed", "Cette prédiction a échoué : aucun feedback possible."
            )
        if repository.get_feedback(session, prediction_id) is not None:
            message = f"Un feedback existe déjà pour la prédiction {prediction_id}."
            raise ApiError(409, "feedback_exists", message)
        saved = repository.save_feedback(session, prediction, payload.actual)
        FEEDBACKS.inc()
        return FeedbackResponse(
            feedback_id=saved.id,
            prediction_id=payload.prediction_id,
            prediction=prediction.prediction,
            actual=saved.actual,
            absolute_error=round(saved.absolute_error, 3),
        )


@router.get("/health", response_model=HealthResponse, tags=["exploitation"])
def health(request: Request) -> JSONResponse:
    """État de service : 200 si base et modèle sont disponibles, 503 (dégradé) sinon."""
    database_ok = is_reachable(request.app.state.engine)
    last_run = None
    if database_ok:
        with _sessions(request)() as session:
            run = repository.latest_pipeline_run(session)
        if run is not None:
            outcome = f" ({run.outcome})" if run.outcome else ""
            last_run = f"{run.state}{outcome} {run.created_at.isoformat(timespec='seconds')}"

    service = _service(request)
    model = service.current
    body = HealthResponse(
        status="ok" if database_ok and model is not None else "degraded",
        database=ComponentStatus(
            ok=database_ok, detail="joignable" if database_ok else "injoignable"
        ),
        model=ComponentStatus(
            ok=model is not None,
            detail=f"{service.model_name} v{model.version}"
            if model
            else (service.last_error or "aucun modèle chargé"),
        ),
        last_pipeline_run=last_run,
    )
    code = status.HTTP_200_OK if body.status == "ok" else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(status_code=code, content=body.model_dump(mode="json"))


@router.get("/health/live", tags=["exploitation"])
def liveness() -> dict[str, str]:
    """Le processus répond (utilisé par le healthcheck Docker)."""
    return {"status": "alive"}


def _model_info(service: ModelService) -> ModelInfo:
    current = service.current
    return ModelInfo(
        model_name=service.model_name,
        alias=service.alias,
        version=current.version if current else None,
        loaded_at=current.loaded_at if current else None,
        last_error=service.last_error,
    )


@router.get("/model", response_model=ModelInfo, tags=["exploitation"])
def model_info(request: Request) -> ModelInfo:
    """Version du modèle actuellement servie."""
    return _model_info(_service(request))


@router.post(
    "/model/reload",
    response_model=ModelInfo,
    responses={503: {"model": ErrorResponse}},
    tags=["exploitation"],
)
def reload_model(request: Request) -> ModelInfo:
    """Recharge la version portant l'alias (après une promotion ou un rollback)."""
    service = _service(request)
    try:
        service.load()
    except Exception as error:
        current = service.current
        kept = f" La version {current.version} reste servie." if current else ""
        raise ApiError(
            503, "model_reload_failed", f"Rechargement impossible : {error}.{kept}"
        ) from error
    return _model_info(service)
