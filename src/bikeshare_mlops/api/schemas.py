"""Contrats d'entrée/sortie de l'API.

Validation stricte : types exacts, plages, champs inconnus refusés.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

MAX_BATCH_SIZE = 1000

_Ratio = Annotated[float, Field(ge=0.0, le=1.0)]
_Binary = Annotated[int, Field(ge=0, le=1)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class RentalFeatures(_Strict):
    """Conditions d'une heure donnée (même codage que le dataset UCI ``hour.csv``)."""

    season: Annotated[int, Field(ge=1, le=4, description="1 hiver, 2 printemps, 3 été, 4 automne")]
    yr: Annotated[int, Field(ge=0, le=1, description="0 = 2011, 1 = 2012")]
    mnth: Annotated[int, Field(ge=1, le=12)]
    hr: Annotated[int, Field(ge=0, le=23)]
    holiday: _Binary
    weekday: Annotated[int, Field(ge=0, le=6, description="0 = dimanche")]
    workingday: _Binary
    weathersit: Annotated[int, Field(ge=1, le=4, description="1 dégagé … 4 très mauvais")]
    temp: _Ratio
    atemp: _Ratio
    hum: _Ratio
    windspeed: _Ratio

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        json_schema_extra={
            "example": {
                "season": 3, "yr": 1, "mnth": 7, "hr": 17, "holiday": 0, "weekday": 3,
                "workingday": 1, "weathersit": 1, "temp": 0.72, "atemp": 0.66,
                "hum": 0.45, "windspeed": 0.19,
            }
        },
    )  # fmt: skip


class BatchPredictionRequest(_Strict):
    instances: Annotated[list[RentalFeatures], Field(min_length=1, max_length=MAX_BATCH_SIZE)]


class PredictionResponse(BaseModel):
    prediction_id: UUID
    prediction: float = Field(description="Nombre de locations prévu pour l'heure")
    model_name: str
    model_version: str
    latency_ms: float


class BatchItem(BaseModel):
    prediction_id: UUID
    prediction: float


class BatchPredictionResponse(BaseModel):
    batch_id: UUID
    model_name: str
    model_version: str
    latency_ms: float
    predictions: list[BatchItem]


class FeedbackRequest(_Strict):
    prediction_id: Annotated[UUID, Field(strict=False)]
    actual: Annotated[float, Field(ge=0, description="Nombre de locations réellement observé")]


class FeedbackResponse(BaseModel):
    feedback_id: int
    prediction_id: UUID
    prediction: float
    actual: float
    absolute_error: float


class ModelInfo(BaseModel):
    model_name: str
    alias: str
    version: str | None
    loaded_at: datetime | None
    last_error: str | None


class ComponentStatus(BaseModel):
    ok: bool
    detail: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    database: ComponentStatus
    model: ComponentStatus
    last_pipeline_run: str | None


class ErrorResponse(BaseModel):
    """Format unique de toutes les erreurs renvoyées par l'API."""

    error: str
    message: str
    details: list[dict[str, str]] | None = None
