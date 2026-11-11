"""Flow Prefect ``batch_prediction`` : prédit un fichier CSV complet avec le modèle champion.

Entrée : CSV au format ``hour.csv`` (les colonnes cibles sont facultatives).
Sortie : même CSV enrichi de ``prediction``, ``model_name`` et ``model_version``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from prefect import flow, get_run_logger, task
from prefect.cache_policies import NO_CACHE
from sklearn.pipeline import Pipeline

from bikeshare_mlops.config import Settings, get_settings
from bikeshare_mlops.data.validate import build_feature_schema, validate_frame
from bikeshare_mlops.features.build import FEATURES
from bikeshare_mlops.models import registry
from bikeshare_mlops.pipelines.hooks import alert_on_failure


@task(name="load-input", cache_policy=NO_CACHE)
def load_input(path: Path, settings: Settings) -> pd.DataFrame:
    """Lit et valide les variables explicatives (mêmes règles qu'à l'entraînement)."""
    frame = pd.read_csv(path)
    validate_frame(frame, f"predict_{path.stem}", settings, schema=build_feature_schema())
    return frame


@task(name="load-champion", cache_policy=NO_CACHE)
def load_champion(settings: Settings) -> tuple[Pipeline, str]:
    registry.configure_tracking(settings)
    return registry.load_by_alias(settings, registry.CHAMPION)


@task(name="predict-and-write", cache_policy=NO_CACHE)
def predict_and_write(
    frame: pd.DataFrame, model: Pipeline, version: str, output: Path, settings: Settings
) -> Path:
    enriched = frame.assign(
        prediction=model.predict(frame[FEATURES]).round(2),
        model_name=settings.mlflow.registered_model_name,
        model_version=version,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output, index=False, lineterminator="\n")
    return output


@flow(name="batch_prediction", on_failure=[alert_on_failure])
def batch_prediction(input_path: str, output_path: str | None = None) -> dict[str, Any]:
    """Prédit toutes les lignes de ``input_path`` ; renvoie le chemin produit et la version."""
    settings = get_settings()
    source = Path(input_path)
    target = (
        Path(output_path)
        if output_path
        else settings.paths.data_dir / "predictions" / f"{source.stem}_predictions.csv"
    )
    frame = load_input(source, settings)
    retry = {
        "retries": settings.orchestration.task_retries,
        "retry_delay_seconds": settings.orchestration.retry_delay_s,
    }
    model, version = load_champion.with_options(**retry)(settings)
    written = predict_and_write(frame, model, version, target, settings)
    summary = {"output": str(written), "rows": len(frame), "model_version": version}
    get_run_logger().info("Prédictions écrites : %s", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    """Exécution ponctuelle (``make predict-batch FILE=...``)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path")
    parser.add_argument("--output-path")
    args = parser.parse_args(argv)
    state = batch_prediction(args.input_path, args.output_path, return_state=True)
    if state.is_failed():
        raise SystemExit(f"Prédiction par lot en échec : {state.message}")
    print(json.dumps(state.result(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
