"""Accès au tracking et au Model Registry MLflow (un seul endroit qui parle à MLflow).

Les alias sont des pointeurs mobiles : ``champion`` = version servie en production,
``challenger`` = dernière version candidate. L'API demande toujours ``models:/<nom>@champion`` :
promouvoir ou revenir en arrière revient à déplacer l'alias, sans redéployer l'API.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from bikeshare_mlops.config import Settings, get_settings

CHAMPION = "champion"
CHALLENGER = "challenger"

TAG_STATUS = "promotion_status"
TAG_REASON = "promotion_reason"


def configure_tracking(settings: Settings) -> str:
    """Pointe MLflow vers le serveur configuré et garantit l'existence de l'expérience."""
    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    mlflow.set_registry_uri(settings.mlflow.tracking_uri)
    experiment = mlflow.get_experiment_by_name(settings.mlflow.experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            settings.mlflow.experiment_name, artifact_location=settings.mlflow.artifact_location
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


def client(settings: Settings) -> MlflowClient:
    """Client MLflow pointant vers le serveur configuré."""
    return MlflowClient(
        tracking_uri=settings.mlflow.tracking_uri, registry_uri=settings.mlflow.tracking_uri
    )


def model_uri(settings: Settings, alias: str) -> str:
    """URI d'une version désignée par alias, ex. ``models:/bikeshare-rf@champion``."""
    return f"models:/{settings.mlflow.registered_model_name}@{alias}"


def version_by_alias(settings: Settings, alias: str) -> str | None:
    """Numéro de la version portant l'alias, ou None si l'alias n'existe pas encore."""
    try:
        entry = client(settings).get_model_version_by_alias(
            settings.mlflow.registered_model_name, alias
        )
    except MlflowException:
        return None
    return str(entry.version)


def set_alias(settings: Settings, alias: str, version: str) -> None:
    client(settings).set_registered_model_alias(
        settings.mlflow.registered_model_name, alias, version
    )


def delete_alias(settings: Settings, alias: str) -> None:
    client(settings).delete_registered_model_alias(settings.mlflow.registered_model_name, alias)


def tag_version(settings: Settings, version: str, tags: dict[str, str]) -> None:
    mlflow_client = client(settings)
    for key, value in tags.items():
        mlflow_client.set_model_version_tag(
            settings.mlflow.registered_model_name, version, key, value
        )


@dataclass(frozen=True)
class VersionRow:
    """Ligne de l'état du registre affiché par ``make models``."""

    version: str
    aliases: str
    status: str
    test_rmse: str
    data_sha256: str


def list_versions(settings: Settings) -> list[VersionRow]:
    """Toutes les versions du modèle enregistré, de la plus récente à la plus ancienne."""
    mlflow_client = client(settings)
    name = settings.mlflow.registered_model_name
    try:
        alias_map = mlflow_client.get_registered_model(name).aliases
    except MlflowException:
        return []
    aliases_by_version: dict[str, list[str]] = {}
    for alias, version in alias_map.items():
        aliases_by_version.setdefault(str(version), []).append(alias)

    versions = mlflow_client.search_model_versions(f"name = '{name}'")
    rows = []
    for item in sorted(versions, key=lambda v: int(v.version), reverse=True):
        rows.append(
            VersionRow(
                version=str(item.version),
                aliases=",".join(sorted(aliases_by_version.get(str(item.version), []))) or "-",
                status=item.tags.get(TAG_STATUS, "-"),
                test_rmse=item.tags.get("test_rmse", "-"),
                data_sha256=item.tags.get("data_sha256", "-")[:12],
            )
        )
    return rows


def main(argv: list[str] | None = None) -> None:
    """Affiche l'état du registre (versions, alias, statut de promotion)."""
    argparse.ArgumentParser(description=main.__doc__).parse_args(argv)
    settings = get_settings()
    rows = list_versions(settings)
    if not rows:
        print(f"Aucune version enregistrée pour {settings.mlflow.registered_model_name}.")
        return
    header = f"{'version':>7}  {'alias':<20} {'statut':<12} {'rmse':>8}  données"
    print(header)
    print("-" * len(header))
    for row in rows:
        rmse = f"{float(row.test_rmse):8.2f}" if row.test_rmse != "-" else f"{'-':>8}"
        print(f"{row.version:>7}  {row.aliases:<20} {row.status:<12} {rmse}  {row.data_sha256}")


if __name__ == "__main__":
    main()
