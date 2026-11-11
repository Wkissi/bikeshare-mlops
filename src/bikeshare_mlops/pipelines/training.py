"""Flow Prefect ``training_pipeline``.

Étapes : ingest → validate → build_dataset → train → register → promote.

- Les lots arrivent dans ``incoming_dir``. Chacun est contrôlé AVANT tout prétraitement.
- Un lot non conforme est mis en quarantaine (``rejected_dir``), son rapport est archivé dans
  MLflow, le flow échoue et une alerte part : aucun modèle n'est entraîné.
- Les lots conformes rejoignent ``processed_dir`` ; le modèle est entraîné sur leur cumul.
- Les tâches qui parlent à MLflow (réseau) ont des retries.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from prefect import flow, get_run_logger, task
from prefect.cache_policies import NO_CACHE
from prefect.runtime import flow_run

from bikeshare_mlops.config import Settings, get_settings
from bikeshare_mlops.data.dataset import load_batches
from bikeshare_mlops.data.ingest import BatchInfo, describe_batch, list_batches
from bikeshare_mlops.data.validate import validate_file
from bikeshare_mlops.exceptions import DataValidationError
from bikeshare_mlops.models import registry
from bikeshare_mlops.models.promote import PromotionDecision, evaluate_and_promote
from bikeshare_mlops.models.train import (
    RegisteredModel,
    TrainingResult,
    log_and_register,
    train_and_evaluate,
)
from bikeshare_mlops.pipelines.hooks import alert_on_failure, record_run
from bikeshare_mlops.utils.hashing import file_sha256


@dataclass(frozen=True)
class BatchOutcome:
    """Résultat du contrôle qualité d'un lot reçu."""

    path: Path
    passed: bool
    report_path: Path
    n_failures: int
    info: BatchInfo | None


def _retry(settings: Settings) -> dict[str, int]:
    return {
        "retries": settings.orchestration.task_retries,
        "retry_delay_seconds": settings.orchestration.retry_delay_s,
    }


@task(name="ingest-incoming", cache_policy=NO_CACHE)
def collect_incoming(settings: Settings) -> list[Path]:
    """Liste les lots arrivés, dans l'ordre chronologique."""
    batches = list_batches(settings.paths.incoming_dir)
    get_run_logger().info("Lots reçus : %s", [path.name for path in batches] or "aucun")
    return batches


@task(name="validate-batch", cache_policy=NO_CACHE)
def check_batch(path: Path, settings: Settings) -> BatchOutcome:
    """Contrôle qualité d'un lot (n'échoue pas : le flow décide de la suite)."""
    try:
        result = validate_file(path, settings)
    except DataValidationError as error:
        get_run_logger().error("Lot %s non conforme : %s", path.name, error)
        return BatchOutcome(path, False, error.report_path, error.n_failures, None)
    return BatchOutcome(path, True, result.report_path, 0, describe_batch(path))


@task(name="log-ingestion", cache_policy=NO_CACHE)
def log_ingestion(outcomes: list[BatchOutcome], settings: Settings) -> None:
    """Un run MLflow par lot : métadonnées (hash, volume, période) et rapport qualité."""
    registry.configure_tracking(settings, settings.mlflow.ingestion_experiment_name)
    for outcome in outcomes:
        with mlflow.start_run(run_name=outcome.path.stem):
            mlflow.set_tags(
                {
                    "batch": outcome.path.stem,
                    "validation_status": "passed" if outcome.passed else "rejected",
                }
            )
            if outcome.info is not None:
                params = outcome.info.as_params()
                params.pop("path")
                mlflow.log_params(params)
            else:
                mlflow.log_param("sha256", file_sha256(outcome.path))
            mlflow.log_metric("n_failures", outcome.n_failures)
            mlflow.log_artifact(str(outcome.report_path), artifact_path="validation")


@task(name="move-batches", cache_policy=NO_CACHE)
def move_batches(paths: list[Path], target_dir: Path) -> list[Path]:
    """Déplace des lots (vers processed ou rejected) ; un homonyme est remplacé."""
    target_dir.mkdir(parents=True, exist_ok=True)
    return [Path(shutil.move(str(path), target_dir / path.name)) for path in paths]


@task(name="build-dataset", cache_policy=NO_CACHE)
def build_dataset(settings: Settings) -> pd.DataFrame:
    """Jeu d'entraînement = cumul de tous les lots validés."""
    frame = load_batches(settings.paths.processed_dir)
    get_run_logger().info("Jeu d'entraînement : %d lignes", len(frame))
    return frame


@task(name="train-model", cache_policy=NO_CACHE)
def train_model(frame: pd.DataFrame, settings: Settings) -> TrainingResult:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return train_and_evaluate(frame, settings, settings.paths.reports_dir / "training" / stamp)


@task(name="register-model", cache_policy=NO_CACHE)
def register_model(result: TrainingResult, settings: Settings) -> RegisteredModel:
    batches = [describe_batch(path) for path in list_batches(settings.paths.processed_dir)]
    return log_and_register(result, settings, batches)


@task(name="promote-model", cache_policy=NO_CACHE)
def promote_model(frame: pd.DataFrame, version: str, settings: Settings) -> PromotionDecision:
    return evaluate_and_promote(settings, frame, version)


@flow(name="training_pipeline", on_failure=[alert_on_failure], log_prints=True)
def training_pipeline(retrain_without_new_data: bool = False) -> dict[str, Any]:
    """Traite les lots reçus puis (ré)entraîne, enregistre et évalue un challenger."""
    settings = get_settings()
    logger = get_run_logger()
    retry = _retry(settings)

    incoming = collect_incoming(settings)
    if not incoming and not retrain_without_new_data:
        logger.info("Aucun nouveau lot : rien à faire.")
        return {"status": "no_new_data"}

    outcomes = [check_batch(path, settings) for path in incoming]
    if outcomes:
        log_ingestion.with_options(**retry)(outcomes, settings)

    rejected = [outcome for outcome in outcomes if not outcome.passed]
    if rejected:
        move_batches([outcome.path for outcome in rejected], settings.paths.rejected_dir)
        first = rejected[0]
        raise DataValidationError(
            ", ".join(outcome.path.stem for outcome in rejected),
            sum(outcome.n_failures for outcome in rejected),
            first.report_path,
        )
    accepted = move_batches([outcome.path for outcome in outcomes], settings.paths.processed_dir)

    frame = build_dataset(settings)
    result = train_model(frame, settings)
    registered = register_model.with_options(**retry)(result, settings)
    decision = promote_model.with_options(**retry)(frame, registered.version, settings)

    summary = {
        "status": "promoted" if decision.promoted else "rejected",
        "batches": [path.name for path in accepted],
        "version": registered.version,
        "run_id": registered.run_id,
        "reason": decision.reason,
        "test_rmse": round(result.metrics.rmse, 3),
    }
    logger.info("Résultat : %s", json.dumps(summary, ensure_ascii=False))
    record_run(
        "training_pipeline",
        str(flow_run.id),
        "COMPLETED",
        outcome=summary["status"],
        model_version=registered.version,
        message=decision.reason,
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    """Exécution ponctuelle du flow depuis la ligne de commande (``make pipeline``)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrain", action="store_true", help="réentraîner même sans nouveau lot")
    args = parser.parse_args(argv)
    state = training_pipeline(retrain_without_new_data=args.retrain, return_state=True)
    if state.is_failed():
        raise SystemExit(f"Pipeline en échec : {state.message}")
    print(json.dumps(state.result(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
