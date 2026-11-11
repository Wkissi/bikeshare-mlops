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
