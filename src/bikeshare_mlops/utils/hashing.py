"""Empreintes de fichiers et de jeux de données (traçabilité des versions de données)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

_CHUNK_SIZE = 1 << 20


def file_sha256(path: Path) -> str:
    """SHA-256 hexadécimal d'un fichier, lu par blocs."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataframe_sha256(frame: pd.DataFrame) -> str:
    """Empreinte stable d'un DataFrame (contenu + noms de colonnes, indépendante de l'index)."""
    digest = hashlib.sha256()
    digest.update(",".join(map(str, frame.columns)).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    return digest.hexdigest()
