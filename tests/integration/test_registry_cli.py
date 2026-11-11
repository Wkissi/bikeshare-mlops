"""Commandes du Makefile (train, promote, models, rollback) contre un MLflow SQLite temporaire."""

from __future__ import annotations

from pathlib import Path

import pytest

from bikeshare_mlops.models import promote, registry, train
from factories import make_hourly_frame

pytestmark = pytest.mark.integration


@pytest.fixture
def batches_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Lots synthétiques ; MLflow SQLite jetable fourni par le garde-fou de conftest."""
    directory = tmp_path / "processed"
    directory.mkdir()
    frame = make_hourly_frame(days=90)
    for month, chunk in frame.groupby(frame["dteday"].str.slice(0, 7)):
        chunk.to_csv(directory / f"batch_{month.replace('-', '_')}.csv", index=False)

    env = {
        "BIKESHARE_MLFLOW__EXPERIMENT_NAME": "cli-exp",
        "BIKESHARE_TRAINING__MODEL__N_ESTIMATORS": "5",
        "BIKESHARE_TRAINING__MODEL__N_JOBS": "1",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return directory


def test_cli_train_promote_list_rollback(
    batches_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    registry.main([])
    assert "Aucune version" in capsys.readouterr().out

    train.main(["--data-dir", str(batches_dir)])
    assert '"challenger_version": "1"' in capsys.readouterr().out

    promote.main(["decide", "--data-dir", str(batches_dir)])
    assert "Version 1 : PROMU" in capsys.readouterr().out

    registry.main([])
    listing = capsys.readouterr().out
    assert "champion" in listing and "promoted" in listing

    with pytest.raises(SystemExit, match="déjà le champion"):
        promote.main(["rollback", "--version", "1"])
