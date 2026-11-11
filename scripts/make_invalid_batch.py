"""Fabrique un lot volontairement non conforme pour démontrer le blocage du pipeline.

Anomalies injectées : colonne ``hum`` supprimée, heures et températures hors plage,
comptages négatifs et incohérents, valeurs nulles et lignes dupliquées.

Usage : uv run python scripts/make_invalid_batch.py [--source FICHIER] [--output FICHIER]
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from bikeshare_mlops.config import get_settings
from bikeshare_mlops.data.ingest import list_batches


def corrupt(frame: pd.DataFrame) -> pd.DataFrame:
    """Applique un échantillon représentatif d'anomalies à une copie du lot."""
    bad = frame.drop(columns=["hum"]).copy()
    bad["weathersit"] = bad["weathersit"].astype("float64")
    bad.loc[bad.index[:5], "hr"] = 30
    bad.loc[bad.index[5:10], "temp"] = 1.7
    bad.loc[bad.index[10:15], "cnt"] = -5
    bad.loc[bad.index[15:20], "weathersit"] = np.nan
    return pd.concat([bad, bad.head(3)], ignore_index=True)


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="lot valide servant de base")
    parser.add_argument("--output", type=Path, help="fichier produit (défaut : incoming_dir)")
    args = parser.parse_args()

    source = args.source
    if source is None:
        batches = list_batches(settings.paths.batches_dir)
        if not batches:
            raise SystemExit("Aucun lot disponible : lancer d'abord `make data`.")
        source = batches[0]

    stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    output = args.output or settings.paths.incoming_dir / f"batch_invalid_{stamp}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    corrupt(pd.read_csv(source)).to_csv(output, index=False, lineterminator="\n")
    print(f"Lot invalide écrit : {output} (base : {source.name})")


if __name__ == "__main__":
    main()
