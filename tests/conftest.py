"""Fixtures partagées : chaque test part d'une configuration propre."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from bikeshare_mlops.config import get_settings


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> Iterator[None]:
    """Vide le cache de configuration avant et après chaque test."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
