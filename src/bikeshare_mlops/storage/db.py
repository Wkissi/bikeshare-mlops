"""Connexion à la base : moteur SQLAlchemy et fabrique de sessions."""

from __future__ import annotations

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from bikeshare_mlops.config import Settings


def build_engine(settings: Settings) -> Engine:
    """Moteur avec vérification des connexions (pre_ping) et délai de connexion court.

    Si PostgreSQL tombe, les requêtes échouent en quelques secondes au lieu de bloquer l'API.
    """
    url = settings.database.url.get_secret_value()
    if url.startswith("sqlite"):
        return create_engine(url)
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_size=settings.database.pool_size,
        connect_args={"connect_timeout": settings.database.connect_timeout_s},
    )


def session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


def is_reachable(engine: Engine) -> bool:
    """True si la base répond à une requête triviale."""
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError:
        return False
    return True
