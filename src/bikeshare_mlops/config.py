"""Chargement de la configuration : YAML versionné + surcharges par variables d'environnement.

Ordre de priorité (du plus fort au plus faible) :
arguments explicites > variables d'environnement > fichier .env > config/config.yaml.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

DEFAULT_CONFIG_FILE = Path("config/config.yaml")
CONFIG_FILE_ENV = "BIKESHARE_CONFIG_FILE"


class _Section(BaseModel):
    """Base des sections : toute clé inconnue est une erreur (faute de frappe détectée tôt)."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class PathsConfig(_Section):
    data_dir: Path
    raw_dir: Path
    batches_dir: Path
    incoming_dir: Path
    processed_dir: Path
    rejected_dir: Path
    reports_dir: Path
    logs_dir: Path


class DataConfig(_Section):
    source_url: str
    archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    csv_name: str = "hour.csv"
    download_timeout_s: float = Field(default=60, gt=0)


class QualityConfig(_Section):
    min_rows: int = Field(default=500, ge=1)
    max_report_failures: int = Field(default=200, ge=1)


class ModelParams(_Section):
    """Hyperparamètres du RandomForestRegressor."""

    n_estimators: int = Field(default=200, ge=1)
    max_depth: int | None = Field(default=None, ge=1)
    min_samples_leaf: int = Field(default=1, ge=1)
    max_features: float = Field(default=1.0, gt=0, le=1)
    n_jobs: int = -1
    random_state: int = 42


class TrainingConfig(_Section):
    test_fraction: float = Field(default=0.2, gt=0, lt=1)
    split_salt: str = "bikeshare"
    min_train_rows: int = Field(default=1000, ge=1)
    model: ModelParams = ModelParams()


class MlflowConfig(_Section):
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "bikeshare-demand"
    ingestion_experiment_name: str = "bikeshare-ingestion"
    registered_model_name: str = "bikeshare-rf"
    artifact_location: str | None = None


class OrchestrationConfig(_Section):
    training_cron: str = "0 6 * * 1"
    watch_interval_s: int = Field(default=60, ge=10)
    task_retries: int = Field(default=3, ge=0)
    retry_delay_s: int = Field(default=10, ge=0)


class AlertsConfig(_Section):
    webhook_url: str | None = None
    file_name: str = "alerts.jsonl"
    timeout_s: float = Field(default=5, gt=0)


class PromotionConfig(_Section):
    """Règles de mise en service d'un challenger."""

    min_improvement_pct: float = Field(default=2.0, ge=0)
    min_gain_vs_baseline_pct: float = Field(default=30.0, ge=0)
    max_segment_degradation_pct: float = Field(default=10.0, ge=0)
    max_single_prediction_ms: float = Field(default=200.0, gt=0)


class LoggingConfig(_Section):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    json_format: bool = True


def config_file() -> Path:
    """Chemin du fichier YAML, surchargeable par la variable BIKESHARE_CONFIG_FILE."""
    return Path(os.environ.get(CONFIG_FILE_ENV, DEFAULT_CONFIG_FILE))


class Settings(BaseSettings):
    """Configuration complète de l'application."""

    model_config = SettingsConfigDict(
        env_prefix="BIKESHARE_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    project_name: str
    paths: PathsConfig
    data: DataConfig
    quality: QualityConfig = QualityConfig()
    training: TrainingConfig = TrainingConfig()
    mlflow: MlflowConfig = MlflowConfig()
    promotion: PromotionConfig = PromotionConfig()
    orchestration: OrchestrationConfig = OrchestrationConfig()
    alerts: AlertsConfig = AlertsConfig()
    logging: LoggingConfig = LoggingConfig()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_source = YamlConfigSettingsSource(
            settings_cls, yaml_file=config_file(), yaml_file_encoding="utf-8"
        )
        return init_settings, env_settings, dotenv_settings, yaml_source


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Instance unique, mise en cache, de la configuration."""
    return Settings()
