"""Exceptions métier : chaque incident attendu a son type, pour être capté et alerté clairement."""

from __future__ import annotations

from pathlib import Path


class BikeshareError(Exception):
    """Racine de toutes les erreurs métier du projet."""


class ChecksumMismatchError(BikeshareError):
    """Le fichier téléchargé ne correspond pas à l'empreinte attendue (corrompu ou modifié)."""

    def __init__(self, path: Path, expected: str, actual: str) -> None:
        super().__init__(f"Checksum invalide pour {path} : attendu {expected}, obtenu {actual}")
        self.path = path
        self.expected = expected
        self.actual = actual


class DataValidationError(BikeshareError):
    """Un lot de données ne respecte pas le contrat qualité : l'entraînement doit s'arrêter."""

    def __init__(self, batch_name: str, n_failures: int, report_path: Path) -> None:
        super().__init__(
            f"Lot {batch_name} non conforme : {n_failures} anomalie(s), rapport : {report_path}"
        )
        self.batch_name = batch_name
        self.n_failures = n_failures
        self.report_path = report_path
