"""Schéma relationnel. Toute modification passe par une migration Alembic (dossier migrations/)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(UTC)


def _created_at() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    """Une ligne par prédiction demandée, réussie ou non (traçabilité complète)."""

    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = _created_at()
    batch_id: Mapped[str | None] = mapped_column(String(36), index=True)
    inputs: Mapped[dict[str, Any]] = mapped_column(JSON)
    prediction: Mapped[float | None] = mapped_column(Float)
    model_name: Mapped[str | None] = mapped_column(String(100))
    model_version: Mapped[str | None] = mapped_column(String(20), index=True)
    latency_ms: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(20), index=True)
    error: Mapped[str | None] = mapped_column(Text)


class Feedback(Base):
    """Valeur réellement observée pour une prédiction (au plus un feedback par prédiction)."""

    __tablename__ = "feedbacks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[str] = mapped_column(
        ForeignKey("predictions.id", ondelete="CASCADE"), unique=True
    )
    created_at: Mapped[datetime] = _created_at()
    actual: Mapped[float] = mapped_column(Float)
    absolute_error: Mapped[float] = mapped_column(Float)


class PipelineRun(Base):
    """Issue de chaque exécution de ``training_pipeline`` (alimente la supervision)."""

    __tablename__ = "pipeline_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = _created_at()
    flow_name: Mapped[str] = mapped_column(String(100))
    flow_run_id: Mapped[str] = mapped_column(String(64))
    state: Mapped[str] = mapped_column(String(20))
    outcome: Mapped[str | None] = mapped_column(String(30))
    model_version: Mapped[str | None] = mapped_column(String(20))
    message: Mapped[str | None] = mapped_column(Text)
