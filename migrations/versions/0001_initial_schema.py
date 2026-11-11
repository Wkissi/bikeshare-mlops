"""Schéma initial : prédictions, feedbacks, runs de pipeline.

Revision ID: 0001
Revises:
Create Date: 2026-11-11
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "predictions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("batch_id", sa.String(36), nullable=True),
        sa.Column("inputs", sa.JSON(), nullable=False),
        sa.Column("prediction", sa.Float(), nullable=True),
        sa.Column("model_name", sa.String(100), nullable=True),
        sa.Column("model_version", sa.String(20), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
    )
    op.create_index("ix_predictions_created_at", "predictions", ["created_at"])
    op.create_index("ix_predictions_batch_id", "predictions", ["batch_id"])
    op.create_index("ix_predictions_model_version", "predictions", ["model_version"])
    op.create_index("ix_predictions_status", "predictions", ["status"])

    op.create_table(
        "feedbacks",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "prediction_id",
            sa.String(36),
            sa.ForeignKey("predictions.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("actual", sa.Float(), nullable=False),
        sa.Column("absolute_error", sa.Float(), nullable=False),
    )
    op.create_index("ix_feedbacks_created_at", "feedbacks", ["created_at"])

    op.create_table(
        "pipeline_runs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("flow_name", sa.String(100), nullable=False),
        sa.Column("flow_run_id", sa.String(64), nullable=False),
        sa.Column("state", sa.String(20), nullable=False),
        sa.Column("outcome", sa.String(30), nullable=True),
        sa.Column("model_version", sa.String(20), nullable=True),
        sa.Column("message", sa.Text(), nullable=True),
    )
    op.create_index("ix_pipeline_runs_created_at", "pipeline_runs", ["created_at"])


def downgrade() -> None:
    op.drop_table("pipeline_runs")
    op.drop_table("feedbacks")
    op.drop_table("predictions")
