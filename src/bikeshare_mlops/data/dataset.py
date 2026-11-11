"""Assemblage du jeu d'entraînement à partir des lots validés et découpage train/test stable.

Le découpage est fait par hash de (date, heure) : un relevé est TOUJOURS du même côté, même quand
de nouveaux lots arrivent. Champion et challenger sont donc comparés sur un jeu de test commun,
et aucune ligne de test d'un ancien modèle ne fuit dans l'entraînement d'un nouveau.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from bikeshare_mlops.data.ingest import list_batches

_HASH_BUCKETS = 10_000


def load_batches(directory: Path) -> pd.DataFrame:
    """Concatène tous les lots d'un dossier, triés chronologiquement."""
    batches = list_batches(directory)
    if not batches:
        raise FileNotFoundError(f"Aucun lot dans {directory}")
    frame = pd.concat((pd.read_csv(path) for path in batches), ignore_index=True)
    return frame.sort_values(["dteday", "hr"], kind="stable").reset_index(drop=True)


def is_test_row(frame: pd.DataFrame, test_fraction: float, salt: str) -> pd.Series:
    """Masque booléen : True si la ligne appartient au jeu de test (déterministe)."""
    keys = salt + ":" + frame["dteday"].astype(str) + ":" + frame["hr"].astype(str)
    buckets = keys.map(
        lambda key: (
            int(hashlib.md5(key.encode("utf-8"), usedforsecurity=False).hexdigest(), 16)
            % _HASH_BUCKETS
        )
    )
    return buckets < int(test_fraction * _HASH_BUCKETS)


def train_test_split_stable(
    frame: pd.DataFrame, test_fraction: float, salt: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sépare un jeu en (train, test) selon ``is_test_row``."""
    mask = is_test_row(frame, test_fraction, salt)
    return frame.loc[~mask].reset_index(drop=True), frame.loc[mask].reset_index(drop=True)
