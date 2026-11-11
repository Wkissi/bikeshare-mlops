"""Ingestion : téléchargement idempotent vérifié, découpage mensuel, arrivée simulée des lots."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import httpx
import pandas as pd
import pytest

from bikeshare_mlops.config import CONFIG_FILE_ENV, Settings
from bikeshare_mlops.data import ingest
from bikeshare_mlops.data.ingest import (
    describe_batch,
    download_archive,
    extract_csv,
    list_batches,
    release_next_batches,
    split_into_batches,
)
from bikeshare_mlops.exceptions import ChecksumMismatchError
from factories import make_hourly_frame

URL = "https://example.test/bike.zip"
REPO_CONFIG = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def _zip_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr("hour.csv", frame.to_csv(index=False))
    return buffer.getvalue()


class _CountingServer:
    """Faux serveur HTTP : renvoie l'archive et compte les appels."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.calls = 0

    def client(self) -> httpx.Client:
        def handler(request: httpx.Request) -> httpx.Response:
            self.calls += 1
            return httpx.Response(200, content=self.payload)

        return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.fixture
def archive_payload() -> bytes:
    return _zip_bytes(make_hourly_frame(days=62))


def test_download_is_verified_and_idempotent(archive_payload: bytes, tmp_path: Path) -> None:
    server = _CountingServer(archive_payload)
    sha = hashlib.sha256(archive_payload).hexdigest()
    target = tmp_path / "raw" / "bike.zip"

    download_archive(URL, target, sha, 5, client=server.client())
    download_archive(URL, target, sha, 5, client=server.client())

    assert target.read_bytes() == archive_payload
    assert server.calls == 1


def test_corrupted_archive_is_redownloaded(archive_payload: bytes, tmp_path: Path) -> None:
    server = _CountingServer(archive_payload)
    target = tmp_path / "bike.zip"
    target.write_bytes(b"corrompu")

    download_archive(URL, target, hashlib.sha256(archive_payload).hexdigest(), 5, server.client())

    assert server.calls == 1
    assert target.read_bytes() == archive_payload


def test_checksum_mismatch_raises_and_leaves_nothing(
    archive_payload: bytes, tmp_path: Path
) -> None:
    target = tmp_path / "bike.zip"
    with pytest.raises(ChecksumMismatchError):
        download_archive(URL, target, "0" * 64, 5, client=_CountingServer(archive_payload).client())
    assert list(tmp_path.iterdir()) == []


def test_http_error_is_propagated(tmp_path: Path) -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(503)))
    with pytest.raises(httpx.HTTPStatusError):
        download_archive(URL, tmp_path / "bike.zip", "0" * 64, 5, client=client)


def test_extract_and_split_by_month(archive_payload: bytes, settings: Settings) -> None:
    archive = settings.paths.raw_dir / "bike.zip"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(archive_payload)

    csv_path = extract_csv(archive, "hour.csv", settings.paths.raw_dir)
    infos = split_into_batches(csv_path, settings.paths.batches_dir)

    assert [info.name for info in infos] == ["batch_2011_01", "batch_2011_02", "batch_2011_03"]
    assert sum(info.n_rows for info in infos) == 62 * 24
    assert infos[0].first_date == "2011-01-01"
    assert infos[0].last_date == "2011-01-31"
    assert infos[0].as_params()["sha256"] == describe_batch(infos[0].path).sha256


def test_release_next_batches_never_replays(settings: Settings) -> None:
    split_csv = settings.paths.raw_dir / "hour.csv"
    split_csv.parent.mkdir(parents=True)
    make_hourly_frame(days=62).to_csv(split_csv, index=False)
    split_into_batches(split_csv, settings.paths.batches_dir)

    first = release_next_batches(settings, count=1)
    settings.paths.processed_dir.mkdir(parents=True)
    first[0].replace(settings.paths.processed_dir / first[0].name)
    rest = release_next_batches(settings, count=5)

    assert [path.name for path in first] == ["batch_2011_01.csv"]
    assert [path.name for path in rest] == ["batch_2011_02.csv", "batch_2011_03.csv"]
    assert release_next_batches(settings, count=1) == []
    assert len(list_batches(settings.paths.incoming_dir)) == 2


def test_list_batches_on_missing_dir(tmp_path: Path) -> None:
    assert list_batches(tmp_path / "absent") == []


def test_cli_prepare_then_release(
    archive_payload: bytes, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chaîne complète en ligne de commande, le réseau étant remplacé par un faux serveur."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(CONFIG_FILE_ENV, str(REPO_CONFIG))
    sha = hashlib.sha256(archive_payload).hexdigest()
    monkeypatch.setenv("BIKESHARE_DATA__ARCHIVE_SHA256", sha)
    server = _CountingServer(archive_payload)
    fake_client = server.client()
    monkeypatch.setattr(ingest.httpx, "Client", lambda **_: fake_client)

    ingest.main(["prepare"])
    ingest.main(["release", "--count", "2"])

    assert server.calls == 1
    assert len(list_batches(tmp_path / "batches_dir")) == 3
    assert len(list_batches(tmp_path / "incoming_dir")) == 2
