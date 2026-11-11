"""Entraînement : découpage stable, apprentissage du pipeline et de la baseline, évaluation,
puis traçage complet dans MLflow et enregistrement de la version comme ``challenger``.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline

from bikeshare_mlops.config import Settings, get_settings
from bikeshare_mlops.data.dataset import load_batches, train_test_split_stable
from bikeshare_mlops.data.ingest import BatchInfo, describe_batch, list_batches
from bikeshare_mlops.exceptions import InsufficientDataError
from bikeshare_mlops.features.build import TARGET, build_pipeline, split_features_target
from bikeshare_mlops.models import registry
from bikeshare_mlops.models.baseline import HourlyMeanBaseline
from bikeshare_mlops.models.evaluate import (
    Metrics,
    compute_metrics,
    feature_importances,
    metrics_by_segment,
    save_diagnostic_plots,
)
from bikeshare_mlops.utils.hashing import dataframe_sha256
from bikeshare_mlops.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

# MLflow sérialise les modèles sklearn avec skops (pas de pickle arbitraire) : seuls les types
# listés ici, relus et maîtrisés, sont autorisés au rechargement.
SKOPS_TRUSTED_TYPES = [
    "bikeshare_mlops.features.build.CyclicalEncoder",
    "sklearn.tree._tree.Tree",
]


@dataclass
class TrainingResult:
    """Tout ce qu'un run produit : modèle, données utilisées, métriques et fichiers."""

    pipeline: Pipeline
    data_sha256: str
    train_rows: int
    test_rows: int
    metrics: Metrics
    baseline_metrics: Metrics
    segment_metrics: dict[str, Metrics]
    test_frame: pd.DataFrame
    input_example: pd.DataFrame
    artifacts: list[Path] = field(default_factory=list)

    def summary(self) -> dict[str, object]:
        """Résumé sérialisable (métriques, volumes, version des données)."""
        return {
            "data_sha256": self.data_sha256,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            **self.metrics.as_dict("test_"),
            **self.baseline_metrics.as_dict("baseline_"),
            "segments": {name: m.as_dict() for name, m in self.segment_metrics.items()},
        }


@dataclass(frozen=True)
class RegisteredModel:
    """Référence d'un modèle tracé : run MLflow et version du registre."""

    run_id: str
    version: str


def train_and_evaluate(frame: pd.DataFrame, settings: Settings, output_dir: Path) -> TrainingResult:
    """Entraîne le pipeline sur des données validées et l'évalue sur le jeu de test stable."""
    config = settings.training
    if len(frame) < config.min_train_rows:
        raise InsufficientDataError(
            f"{len(frame)} lignes disponibles, minimum requis : {config.min_train_rows}"
        )

    train, test = train_test_split_stable(frame, config.test_fraction, config.split_salt)
    x_train, y_train = split_features_target(train)
    x_test, y_test = split_features_target(test)

    pipeline = build_pipeline(config.model).fit(x_train, y_train)
    # Entraînement parallèle, mais inférence mono-thread : une requête API ne doit pas
    # lancer un pool de threads (latence et contention entre workers uvicorn).
    pipeline.named_steps["model"].set_params(n_jobs=1)
    baseline = HourlyMeanBaseline().fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)

    result = TrainingResult(
        pipeline=pipeline,
        data_sha256=dataframe_sha256(frame),
        train_rows=len(train),
        test_rows=len(test),
        metrics=compute_metrics(y_test, y_pred),
        baseline_metrics=compute_metrics(y_test, baseline.predict(x_test)),
        segment_metrics=metrics_by_segment(test, y_pred, TARGET),
        test_frame=test,
        input_example=x_train.head(5),
    )
    result.artifacts = save_diagnostic_plots(
        test, y_pred, TARGET, feature_importances(pipeline), output_dir
    )
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result.summary(), indent=2), encoding="utf-8")
    result.artifacts.append(metrics_path)

    logger.info(
        "modèle entraîné",
        extra={
            "train_rows": result.train_rows,
            "test_rows": result.test_rows,
            **result.metrics.as_dict("test_"),
            **result.baseline_metrics.as_dict("baseline_"),
        },
    )
    return result


def _serving_requirements() -> list[str]:
    """Dépendances d'inférence déclarées explicitement (versions réellement installées).

    Évite l'inférence automatique de MLflow, qui recharge le modèle dans un sous-processus.
    """
    packages = ("scikit-learn", "skops", "pandas", "numpy")
    return [f"{package}=={metadata.version(package)}" for package in packages]


def log_and_register(
    result: TrainingResult, settings: Settings, batches: list[BatchInfo]
) -> RegisteredModel:
    """Trace le run (paramètres, métriques, données, artefacts) et enregistre le ``challenger``."""
    registry.configure_tracking(settings)
    segment_metrics = {
        f"season_{name}_{metric}": value
        for name, metrics in result.segment_metrics.items()
        for metric, value in metrics.as_dict().items()
    }
    with mlflow.start_run(run_name=f"train-{datetime.now(UTC):%Y%m%dT%H%M%SZ}") as run:
        mlflow.set_tags(
            {
                "data_sha256": result.data_sha256,
                "n_batches": str(len(batches)),
                "last_batch": batches[-1].name if batches else "",
                # Commit Git du code (exporté par le Makefile, figé dans l'image Docker).
                "code_version": os.environ.get("CODE_VERSION", "unknown"),
            }
        )
        mlflow.log_params(
            {f"model.{key}": value for key, value in settings.training.model.model_dump().items()}
        )
        mlflow.log_params(
            {
                "test_fraction": settings.training.test_fraction,
                "split_salt": settings.training.split_salt,
                "train_rows": result.train_rows,
                "test_rows": result.test_rows,
            }
        )
        mlflow.log_metrics(
            result.metrics.as_dict("test_")
            | result.baseline_metrics.as_dict("baseline_")
            | segment_metrics
        )
        mlflow.log_input(
            mlflow.data.from_pandas(result.test_frame, name="test_set", targets=TARGET),
            context="test",
        )
        mlflow.log_dict({"batches": [b.as_params() for b in batches]}, "data/batches.json")
        for path in result.artifacts:
            mlflow.log_artifact(str(path), artifact_path="evaluation")
        with warnings.catch_warnings():
            # Colonnes entières voulues : l'API valide les types en amont (pas de valeur manquante).
            warnings.filterwarnings("ignore", message=".*Inferred schema contains integer")
            signature = infer_signature(
                result.input_example, result.pipeline.predict(result.input_example)
            )
            info = mlflow.sklearn.log_model(
                result.pipeline,
                name="model",
                signature=signature,
                input_example=result.input_example,
                registered_model_name=settings.mlflow.registered_model_name,
                skops_trusted_types=SKOPS_TRUSTED_TYPES,
                pip_requirements=_serving_requirements(),
            )

    version = str(info.registered_model_version)
    registry.tag_version(
        settings,
        version,
        {
            "data_sha256": result.data_sha256,
            "run_id": run.info.run_id,
            "test_rmse": f"{result.metrics.rmse:.4f}",
            registry.TAG_STATUS: "pending",
        },
    )
    registry.set_alias(settings, registry.CHALLENGER, version)
    logger.info(
        "challenger enregistré",
        extra={"version": version, "run_id": run.info.run_id, "test_rmse": result.metrics.rmse},
    )
    return RegisteredModel(run_id=run.info.run_id, version=version)


def main(argv: list[str] | None = None) -> None:
    """Entraîne sur un dossier de lots ; trace dans MLflow sauf avec ``--no-mlflow``."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=settings.paths.processed_dir)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-mlflow", action="store_true", help="évaluation locale seulement")
    args = parser.parse_args(argv)

    configure_logging(settings.logging.level, settings.logging.json_format)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or settings.paths.reports_dir / "training" / stamp
    result = train_and_evaluate(load_batches(args.data_dir), settings, output_dir)
    summary = result.summary()
    if not args.no_mlflow:
        batches = [describe_batch(path) for path in list_batches(args.data_dir)]
        registered = log_and_register(result, settings, batches)
        summary |= {"run_id": registered.run_id, "challenger_version": registered.version}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
