"""Déploiements Prefect servis par le conteneur ``pipeline-worker``.

- ``watch-incoming`` : toutes les ``watch_interval_s`` secondes, lance ``training_pipeline`` si un
  lot est arrivé dans ``incoming_dir`` (déclenchement à l'arrivée des données) ;
- ``weekly-retrain`` : cron, réentraîne sur toutes les données validées, même sans nouveau lot ;
- ``on-demand`` : prédiction par lot, déclenchée depuis l'UI ou ``prefect deployment run``.

``limit=1`` : un seul flow à la fois, deux entraînements ne se marchent jamais dessus.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from prefect import flow, get_run_logger, serve

from bikeshare_mlops.config import get_settings
from bikeshare_mlops.data.ingest import list_batches
from bikeshare_mlops.pipelines.batch_prediction import batch_prediction
from bikeshare_mlops.pipelines.training import training_pipeline
from bikeshare_mlops.utils.logging import configure_logging


@flow(name="watch_incoming")
def watch_incoming() -> dict[str, Any]:
    """Déclenche l'entraînement seulement si de nouveaux lots attendent."""
    pending = list_batches(get_settings().paths.incoming_dir)
    if not pending:
        get_run_logger().info("Aucun lot en attente.")
        return {"status": "idle"}
    return training_pipeline()


def main() -> None:
    settings = get_settings()
    configure_logging(settings.logging.level, settings.logging.json_format)
    orchestration = settings.orchestration
    serve(
        watch_incoming.to_deployment(
            name="watch-incoming", interval=timedelta(seconds=orchestration.watch_interval_s)
        ),
        training_pipeline.to_deployment(
            name="weekly-retrain",
            cron=orchestration.training_cron,
            parameters={"retrain_without_new_data": True},
        ),
        batch_prediction.to_deployment(name="on-demand"),
        limit=1,
    )


if __name__ == "__main__":
    main()
