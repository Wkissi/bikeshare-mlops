"""Application FastAPI (fabrique ``create_app``, utilisée par uvicorn avec ``--factory``)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sklearn.pipeline import Pipeline
from sqlalchemy import Engine

from bikeshare_mlops import __version__
from bikeshare_mlops.api.errors import install_error_handlers
from bikeshare_mlops.api.model_service import ModelService
from bikeshare_mlops.api.routes import router
from bikeshare_mlops.config import Settings, get_settings
from bikeshare_mlops.models import registry
from bikeshare_mlops.storage.db import build_engine, session_factory
from bikeshare_mlops.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def registry_loader(settings: Settings) -> tuple[Pipeline, str]:
    """Charge depuis MLflow la version portant l'alias configuré (``champion`` par défaut)."""
    registry.configure_tracking(settings)
    return registry.load_by_alias(settings, settings.api.model_alias)


def registry_version(settings: Settings) -> str | None:
    """Version actuellement désignée par l'alias dans le registre (requête légère)."""
    return registry.version_by_alias(settings, settings.api.model_alias)


def create_app(
    settings: Settings | None = None,
    model_service: ModelService | None = None,
    engine: Engine | None = None,
) -> FastAPI:
    """Assemble l'application ; modèle et base sont injectables (tests)."""
    settings = settings or get_settings()
    configure_logging(settings.logging.level, settings.logging.json_format)
    service = model_service or ModelService(
        settings.mlflow.registered_model_name,
        settings.api.model_alias,
        lambda: registry_loader(settings),
        lambda: registry_version(settings),
    )
    db_engine = engine or build_engine(settings)

    async def follow_alias() -> None:
        while True:
            await asyncio.sleep(settings.api.model_refresh_s)
            await asyncio.to_thread(service.refresh)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Démarrage tolérant : sans registre joignable, l'API démarre en mode dégradé
        # (/health le signale) puis charge le modèle dès que le registre revient.
        if service.current is None:
            try:
                service.load()
            except Exception:
                logger.warning("API démarrée sans modèle : mode dégradé")
        follower = asyncio.create_task(follow_alias())
        yield
        follower.cancel()
        db_engine.dispose()

    app = FastAPI(
        title="Bikeshare : prévision des locations horaires",
        version=__version__,
        description="Prédictions unitaires et par lot, feedbacks, état du service et du modèle.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.model_service = service
    app.state.engine = db_engine
    app.state.sessions = session_factory(db_engine)
    install_error_handlers(app)
    app.include_router(router)
    return app
