"""Service de modèle : chargement, prédiction bornée, suivi de l'alias du registre."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from bikeshare_mlops.api.model_service import ModelService, ModelUnavailableError
from bikeshare_mlops.features.build import FEATURES

ROW: dict[str, Any] = dict.fromkeys(FEATURES, 1)


class _Constant:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        assert list(frame.columns) == FEATURES
        return np.full(len(frame), self.value)


class _Registry:
    """Faux registre : la version de l'alias est modifiable, les chargements sont comptés."""

    def __init__(self, version: str | None = "1") -> None:
        self.version = version
        self.loads = 0
        self.down = False

    def loader(self) -> tuple[Any, str]:
        if self.down:
            raise ConnectionError("registre injoignable")
        self.loads += 1
        return _Constant(float(self.version or 0)), self.version or "0"

    def lookup(self) -> str | None:
        if self.down:
            raise ConnectionError("registre injoignable")
        return self.version


def _service(registry: _Registry) -> ModelService:
    return ModelService("m", "champion", registry.loader, registry.lookup)


def test_predict_requires_a_loaded_model() -> None:
    service = _service(_Registry())
    with pytest.raises(ModelUnavailableError):
        service.predict([ROW])


def test_predictions_are_never_negative() -> None:
    service = ModelService("m", "champion", lambda: (_Constant(-5.0), "1"))
    service.load()
    values, version = service.predict([ROW, ROW])
    assert values.tolist() == [0.0, 0.0]
    assert version == "1"


def test_refresh_follows_alias_changes() -> None:
    registry = _Registry("1")
    service = _service(registry)
    service.load()

    assert service.refresh() is False  # alias inchangé : pas de rechargement
    registry.version = "2"
    assert service.refresh() is True
    values, version = service.predict([ROW])
    assert (values.tolist(), version) == ([2.0], "2")
    assert registry.loads == 2


def test_refresh_keeps_model_when_registry_is_down() -> None:
    registry = _Registry("1")
    service = _service(registry)
    service.load()
    registry.down = True

    assert service.refresh() is False
    assert service.current is not None and service.current.version == "1"


def test_refresh_loads_model_once_registry_comes_back() -> None:
    registry = _Registry("3")
    registry.down = True
    service = _service(registry)
    with pytest.raises(ConnectionError):
        service.load()
    assert service.last_error is not None

    registry.down = False
    assert service.refresh() is True
    assert service.current is not None and service.current.version == "3"
    assert service.last_error is None


def test_refresh_without_lookup_or_alias_does_nothing() -> None:
    assert ModelService("m", "champion", _Registry().loader).refresh() is False
    assert _service(_Registry(version=None)).refresh() is False


def test_refresh_survives_failed_load() -> None:
    registry = _Registry("1")
    service = _service(registry)
    service.load()
    registry.version = "2"

    def corrupted() -> tuple[Any, str]:
        raise RuntimeError("artefact corrompu")

    service._loader = corrupted

    assert service.refresh() is False
    assert service.current is not None and service.current.version == "1"
