"""Les migrations Alembic produisent exactement le schéma déclaré par les modèles ORM."""

from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, inspect

from bikeshare_mlops.storage.tables import Base

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]


def test_upgrade_matches_orm_and_downgrade_cleans(tmp_path: Path) -> None:
    url = f"sqlite:///{(tmp_path / 'migrated.db').as_posix()}"
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", url)

    command.upgrade(config, "head")
    engine = create_engine(url)
    with engine.connect() as connection:
        differences = compare_metadata(MigrationContext.configure(connection), Base.metadata)
    assert differences == []
    assert {"predictions", "feedbacks", "pipeline_runs"} <= set(inspect(engine).get_table_names())

    command.downgrade(config, "base")
    assert set(inspect(engine).get_table_names()) == {"alembic_version"}
    engine.dispose()
