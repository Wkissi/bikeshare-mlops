"""Ingestion : téléchargement vérifié de la source UCI, découpage en lots mensuels, arrivée simulée.

Simulation de flux : le dataset (2011-2012) est découpé en 24 lots mensuels dans ``batches_dir``.
« Faire arriver » un lot revient à le copier dans ``incoming_dir``, où le pipeline le récupère.
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd

from bikeshare_mlops.config import Settings, get_settings
from bikeshare_mlops.exceptions import ChecksumMismatchError
from bikeshare_mlops.utils.hashing import file_sha256
from bikeshare_mlops.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

ARCHIVE_NAME = "bike_sharing_dataset.zip"
BATCH_PREFIX = "batch_"


@dataclass(frozen=True)
class BatchInfo:
    """Métadonnées d'un lot, enregistrées pour tracer quelles données ont servi à quoi."""

    name: str
    path: Path
    n_rows: int
    sha256: str
    first_date: str
    last_date: str
    described_at: str

    def as_params(self) -> dict[str, str | int]:
        """Version sérialisable (pour MLflow ou les logs)."""
        data = asdict(self)
        data["path"] = str(self.path)
        return data


def download_archive(
    url: str,
    destination: Path,
    expected_sha256: str,
    timeout_s: float,
    client: httpx.Client | None = None,
) -> Path:
    """Télécharge l'archive si absente ou altérée, puis vérifie son SHA-256.

    Idempotent : une archive déjà présente et intègre n'est pas retéléchargée.
    """
    if destination.exists() and file_sha256(destination) == expected_sha256:
        logger.info("archive déjà présente et intègre", extra={"path": str(destination)})
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    owns_client = client is None
    http = client or httpx.Client(timeout=timeout_s, follow_redirects=True)
    try:
        with http.stream("GET", url) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)
    finally:
        if owns_client:
            http.close()

    actual = file_sha256(partial)
    if actual != expected_sha256:
        partial.unlink()
        raise ChecksumMismatchError(destination, expected_sha256, actual)
    partial.replace(destination)
    logger.info("archive téléchargée et vérifiée", extra={"path": str(destination), "url": url})
    return destination


def extract_csv(archive: Path, csv_name: str, raw_dir: Path) -> Path:
    """Extrait le CSV horaire de l'archive vérifiée (écrase une éventuelle copie locale)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / csv_name
    with zipfile.ZipFile(archive) as bundle, bundle.open(csv_name) as source:
        target.write_bytes(source.read())
    return target


def describe_batch(path: Path) -> BatchInfo:
    """Calcule les métadonnées d'un fichier de lot."""
    frame = pd.read_csv(path)
    dates = frame["dteday"].dropna().astype(str) if "dteday" in frame else pd.Series(dtype=str)
    return BatchInfo(
        name=path.stem,
        path=path,
        n_rows=len(frame),
        sha256=file_sha256(path),
        first_date=str(dates.min()) if len(dates) else "",
        last_date=str(dates.max()) if len(dates) else "",
        described_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


def split_into_batches(csv_path: Path, batches_dir: Path) -> list[BatchInfo]:
    """Découpe le CSV en un fichier par mois : ``batch_YYYY_MM.csv`` (déterministe)."""
    frame = pd.read_csv(csv_path)
    batches_dir.mkdir(parents=True, exist_ok=True)
    month_keys = frame["dteday"].astype(str).str.slice(0, 7)
    infos: list[BatchInfo] = []
    for month, chunk in frame.groupby(month_keys, sort=True):
        path = batches_dir / f"{BATCH_PREFIX}{str(month).replace('-', '_')}.csv"
        chunk.to_csv(path, index=False, lineterminator="\n")
        infos.append(describe_batch(path))
    logger.info("lots mensuels générés", extra={"count": len(infos), "dir": str(batches_dir)})
    return infos


def list_batches(directory: Path) -> list[Path]:
    """Fichiers de lots d'un dossier, triés chronologiquement par nom."""
    if not directory.exists():
        return []
    return sorted(directory.glob(f"{BATCH_PREFIX}*.csv"))


def release_next_batches(settings: Settings, count: int = 1) -> list[Path]:
    """Simule l'arrivée de ``count`` nouveaux lots en les copiant dans ``incoming_dir``.

    Un lot déjà arrivé (présent dans incoming, processed ou rejected) n'est pas rejoué.
    """
    paths = settings.paths
    already_seen = {
        batch.name
        for directory in (paths.incoming_dir, paths.processed_dir, paths.rejected_dir)
        for batch in list_batches(directory)
    }
    pending = [batch for batch in list_batches(paths.batches_dir) if batch.name not in already_seen]
    paths.incoming_dir.mkdir(parents=True, exist_ok=True)
    released = []
    for batch in pending[:count]:
        target = paths.incoming_dir / batch.name
        shutil.copy2(batch, target)
        released.append(target)
    logger.info(
        "lots déposés dans incoming",
        extra={
            "released": [path.name for path in released],
            "remaining": len(pending) - len(released),
        },
    )
    return released


def prepare_source(settings: Settings) -> list[BatchInfo]:
    """Chaîne complète : téléchargement vérifié → extraction → découpage en lots."""
    archive = download_archive(
        url=settings.data.source_url,
        destination=settings.paths.raw_dir / ARCHIVE_NAME,
        expected_sha256=settings.data.archive_sha256,
        timeout_s=settings.data.download_timeout_s,
    )
    csv_path = extract_csv(archive, settings.data.csv_name, settings.paths.raw_dir)
    return split_into_batches(csv_path, settings.paths.batches_dir)


def main(argv: list[str] | None = None) -> None:
    """Point d'entrée en ligne de commande (utilisé par le Makefile)."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="télécharge, vérifie et découpe la source")
    release = commands.add_parser("release", help="fait arriver les prochains lots")
    release.add_argument("--count", type=int, default=1)
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.logging.level, settings.logging.json_format)
    if args.command == "prepare":
        prepare_source(settings)
    else:
        release_next_batches(settings, args.count)


if __name__ == "__main__":
    main()
