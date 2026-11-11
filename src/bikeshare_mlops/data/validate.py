"""Contrôles qualité (pandera) appliqués à chaque lot AVANT tout prétraitement.

Le schéma est le « contrat » des données brutes : colonnes, types, plages, nulls, doublons,
cohérence et volume minimum. Toutes les anomalies sont collectées (mode ``lazy``), écrites dans un
rapport JSON, puis une ``DataValidationError`` interrompt la chaîne.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

from bikeshare_mlops.config import Settings, get_settings
from bikeshare_mlops.exceptions import DataValidationError
from bikeshare_mlops.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

DATE_FORMAT = "%Y-%m-%d"


def _int_column(low: int, high: int | None = None) -> pa.Column:
    check = pa.Check.ge(low) if high is None else pa.Check.in_range(low, high)
    return pa.Column("int64", check, nullable=False)


def _ratio_column() -> pa.Column:
    return pa.Column("float64", pa.Check.in_range(0.0, 1.0), nullable=False, coerce=True)


def _is_iso_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format=DATE_FORMAT, errors="coerce").notna()


def _counts_are_consistent(frame: pd.DataFrame) -> pd.Series | bool:
    required = {"cnt", "casual", "registered"}
    if not required.issubset(frame.columns):
        return True  # l'absence de colonne est déjà signalée par le schéma
    return frame["cnt"] == frame["casual"] + frame["registered"]


def build_schema(min_rows: int) -> pa.DataFrameSchema:
    """Schéma des lots bruts au format ``hour.csv`` (variables météo normalisées dans [0, 1])."""
    return pa.DataFrameSchema(
        columns={
            "instant": _int_column(1),
            "dteday": pa.Column(checks=pa.Check(_is_iso_date, name="iso_date"), nullable=False),
            "season": _int_column(1, 4),
            "yr": _int_column(0, 1),
            "mnth": _int_column(1, 12),
            "hr": _int_column(0, 23),
            "holiday": _int_column(0, 1),
            "weekday": _int_column(0, 6),
            "workingday": _int_column(0, 1),
            "weathersit": _int_column(1, 4),
            "temp": _ratio_column(),
            "atemp": _ratio_column(),
            "hum": _ratio_column(),
            "windspeed": _ratio_column(),
            "casual": _int_column(0),
            "registered": _int_column(0),
            "cnt": _int_column(0),
        },
        checks=[
            pa.Check(lambda frame: len(frame) >= min_rows, name=f"min_rows_{min_rows}"),
            pa.Check(_counts_are_consistent, name="cnt_equals_casual_plus_registered"),
        ],
        unique=["dteday", "hr"],
        report_duplicates="all",
        strict=True,
        name="hourly_rentals",
    )


@dataclass(frozen=True)
class ValidationResult:
    """Résultat d'une validation réussie : données typées + rapport sauvegardé."""

    batch_name: str
    data: pd.DataFrame
    report_path: Path


def _deduplicated_failures(error: SchemaErrors) -> pd.DataFrame:
    """Une anomalie par ligne et par contrôle (pandera répète les contrôles globaux par colonne)."""
    cases = error.failure_cases
    frame_level = (cases["schema_context"] == "DataFrameSchema") & cases["index"].notna()
    per_row = cases[frame_level].drop_duplicates(subset=["check", "index"]).assign(column="<ligne>")
    return pd.concat([cases[~frame_level], per_row], ignore_index=True)


def _failure_records(cases: pd.DataFrame, limit: int) -> tuple[list[dict[str, Any]], Counter[str]]:
    labels = cases["column"].astype(str) + ":" + cases["check"].astype(str)
    by_check: Counter[str] = Counter(labels)
    records = (
        cases[["column", "check", "failure_case", "index"]]
        .head(limit)
        .astype(str)
        .to_dict(orient="records")
    )
    return records, by_check


def _write_report(report: dict[str, Any], reports_dir: Path) -> Path:
    target_dir = reports_dir / "validation"
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    path = target_dir / f"{report['batch']}_{stamp}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def validate_frame(frame: pd.DataFrame, batch_name: str, settings: Settings) -> ValidationResult:
    """Valide un lot ; lève ``DataValidationError`` (rapport à l'appui) s'il n'est pas conforme."""
    schema = build_schema(settings.quality.min_rows)
    report: dict[str, Any] = {
        "batch": batch_name,
        "validated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_rows": len(frame),
        "columns": list(map(str, frame.columns)),
    }
    try:
        validated = schema.validate(frame, lazy=True)
    except SchemaErrors as error:
        cases = _deduplicated_failures(error)
        records, by_check = _failure_records(cases, settings.quality.max_report_failures)
        report |= {
            "status": "failed",
            "n_failures": len(cases),
            "failures_by_check": dict(by_check.most_common()),
            "failures": records,
        }
        path = _write_report(report, settings.paths.reports_dir)
        logger.error(
            "lot rejeté par les contrôles qualité",
            extra={"batch": batch_name, "n_failures": report["n_failures"], "report": str(path)},
        )
        raise DataValidationError(batch_name, report["n_failures"], path) from error

    report |= {"status": "passed", "n_failures": 0}
    path = _write_report(report, settings.paths.reports_dir)
    logger.info("lot conforme", extra={"batch": batch_name, "n_rows": len(frame)})
    return ValidationResult(batch_name=batch_name, data=validated, report_path=path)


def validate_file(path: Path, settings: Settings) -> ValidationResult:
    """Lit puis valide un fichier de lot ; un CSV illisible est lui aussi une non-conformité."""
    try:
        frame = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
        report = {
            "batch": path.stem,
            "validated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "status": "failed",
            "n_failures": 1,
            "failures_by_check": {"file:readable_csv": 1},
            "failures": [{"column": "file", "check": "readable_csv", "failure_case": str(error)}],
        }
        report_path = _write_report(report, settings.paths.reports_dir)
        raise DataValidationError(path.stem, 1, report_path) from error
    return validate_frame(frame, path.stem, settings)


def main(argv: list[str] | None = None) -> None:
    """Valide un ou plusieurs fichiers ; code de sortie 1 si l'un d'eux est non conforme."""
    parser = argparse.ArgumentParser(description="Contrôle qualité de fichiers de lots")
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.logging.level, settings.logging.json_format)
    failures = 0
    for path in args.files:
        try:
            validate_file(path, settings)
        except DataValidationError as error:
            failures += 1
            print(error, file=sys.stderr)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
