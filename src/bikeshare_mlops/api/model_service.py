"""Modèle servi par l'API : chargé depuis le registre, rechargeable à chaud sans interruption."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from bikeshare_mlops.exceptions import BikeshareError
from bikeshare_mlops.features.build import FEATURES
from bikeshare_mlops.utils.logging import get_logger

logger = get_logger(__name__)

ModelLoader = Callable[[], tuple[Pipeline, str]]
VersionLookup = Callable[[], str | None]


class ModelUnavailableError(BikeshareError):
    """Aucun modèle chargé : l'API ne peut pas prédire."""


@dataclass(frozen=True)
class LoadedModel:
    pipeline: Pipeline
    version: str
    loaded_at: datetime


class ModelService:
    """Détient le modèle courant ; un rechargement raté conserve le modèle déjà servi."""

    def __init__(
        self,
        model_name: str,
        alias: str,
        loader: ModelLoader,
        version_lookup: VersionLookup | None = None,
        on_loaded: Callable[[str], None] | None = None,
    ) -> None:
        self.model_name = model_name
        self.alias = alias
        self._loader = loader
        self._version_lookup = version_lookup
        self._on_loaded = on_loaded
        self._current: LoadedModel | None = None
        self._lock = threading.Lock()
        self.last_error: str | None = None

    @property
    def current(self) -> LoadedModel | None:
        return self._current

    def load(self) -> LoadedModel:
        """Charge la version portant l'alias ; lève l'erreur d'origine en cas d'échec."""
        with self._lock:
            try:
                pipeline, version = self._loader()
            except Exception as error:
                self.last_error = f"{type(error).__name__}: {error}"
                logger.error(
                    "chargement du modèle impossible",
                    extra={"alias": self.alias, "error": self.last_error},
                )
                raise
            self._current = LoadedModel(pipeline, version, datetime.now(UTC))
            self.last_error = None
        if self._on_loaded is not None:
            self._on_loaded(version)
        logger.info("modèle chargé", extra={"alias": self.alias, "version": version})
        return self._current

    def refresh(self) -> bool:
        """Recharge si l'alias pointe sur une autre version (promotion, rollback) ; True si changé.

        Appelé périodiquement par chaque worker uvicorn : tous convergent vers la même version.
        Un registre injoignable n'interrompt pas le service : le modèle courant reste servi.
        """
        if self._version_lookup is None:
            return False
        try:
            target = self._version_lookup()
        except Exception as error:
            logger.warning("registre injoignable, modèle conservé", extra={"error": str(error)})
            return False
        current = self._current
        if target is None or (current is not None and current.version == target):
            return False
        try:
            self.load()
        except Exception:
            return False
        return True

    def predict(self, rows: list[dict[str, float | int]]) -> tuple[np.ndarray, str]:
        """Prédictions (>= 0) et version du modèle utilisé."""
        model = self._current
        if model is None:
            raise ModelUnavailableError(
                f"Aucun modèle '{self.alias}' chargé"
                + (f" ({self.last_error})" if self.last_error else "")
            )
        frame = pd.DataFrame.from_records(rows, columns=FEATURES)
        return np.clip(model.pipeline.predict(frame), 0, None), model.version
