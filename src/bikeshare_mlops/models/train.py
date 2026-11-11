"""Entraînement : découpage stable, apprentissage du pipeline et de la baseline, évaluation."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from bikeshare_mlops.config import Settings, get_settings
from bikeshare_mlops.data.dataset import load_batches, train_test_split_stable
from bikeshare_mlops.exceptions import InsufficientDataError
from bikeshare_mlops.features.build import TARGET, build_pipeline, split_features_target
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


def main(argv: list[str] | None = None) -> None:
    """Entraîne sur un dossier de lots et écrit métriques + graphiques dans ``reports/training``."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=settings.paths.processed_dir)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)

    configure_logging(settings.logging.level, settings.logging.json_format)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or settings.paths.reports_dir / "training" / stamp
    result = train_and_evaluate(load_batches(args.data_dir), settings, output_dir)
    print(json.dumps(result.summary(), indent=2))


if __name__ == "__main__":
    main()
