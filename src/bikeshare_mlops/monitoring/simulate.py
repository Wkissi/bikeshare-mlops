"""Simulation de trafic : rejoue des heures réelles contre l'API, puis renvoie la valeur observée.

Sert aux démonstrations et à alimenter le dashboard : prédictions unitaires et par lot, une part
de requêtes volontairement invalides (422), et des feedbacks avec la vraie valeur ``cnt``.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from bikeshare_mlops.config import get_settings
from bikeshare_mlops.data.ingest import list_batches
from bikeshare_mlops.features.build import FEATURES, TARGET


@dataclass
class TrafficReport:
    """Bilan d'une simulation : codes HTTP obtenus, feedbacks envoyés, erreur moyenne."""

    status_codes: Counter[int] = field(default_factory=Counter)
    feedbacks: int = 0
    absolute_errors: list[float] = field(default_factory=list)

    @property
    def mae(self) -> float | None:
        if not self.absolute_errors:
            return None
        return sum(self.absolute_errors) / len(self.absolute_errors)

    def summary(self) -> str:
        codes = ", ".join(f"{code}: {count}" for code, count in sorted(self.status_codes.items()))
        mae = "-" if self.mae is None else f"{self.mae:.1f}"
        return f"réponses [{codes}] ; feedbacks : {self.feedbacks} ; MAE des feedbacks : {mae}"


def sample_hours(source: Path, count: int, seed: int = 0) -> list[tuple[dict[str, Any], float]]:
    """Tire ``count`` heures réelles d'un lot : (variables explicatives, valeur observée)."""
    frame = pd.read_csv(source)
    picked = frame.sample(n=count, replace=count > len(frame), random_state=seed)
    # to_dict convertit en types Python natifs (int, float), attendus par la validation stricte.
    features = picked[FEATURES].to_dict(orient="records")
    return list(zip(features, picked[TARGET].astype(float).tolist(), strict=True))


def send_traffic(
    client: httpx.Client,
    hours: list[tuple[dict[str, Any], float]],
    feedback_ratio: float = 0.8,
    invalid_ratio: float = 0.0,
    batch_size: int = 0,
    seed: int = 0,
) -> TrafficReport:
    """Envoie les heures une par une (ou par lots si ``batch_size`` > 0) puis les feedbacks."""
    rng = random.Random(seed)
    report = TrafficReport()
    pending: list[tuple[str, float]] = []
    step = batch_size or 1
    for start in range(0, len(hours), step):
        chunk = hours[start : start + step]
        if rng.random() < invalid_ratio:
            features = {**chunk[0][0], "hr": 99}  # hors plage : 422 attendu
            report.status_codes[client.post("/predict", json=features).status_code] += 1
            continue
        if batch_size:
            response = client.post("/predict/batch", json={"instances": [h for h, _ in chunk]})
            report.status_codes[response.status_code] += 1
            if response.status_code == 200:
                ids = [item["prediction_id"] for item in response.json()["predictions"]]
                pending.extend(zip(ids, (actual for _, actual in chunk), strict=True))
        else:
            response = client.post("/predict", json=chunk[0][0])
            report.status_codes[response.status_code] += 1
            if response.status_code == 200:
                pending.append((response.json()["prediction_id"], chunk[0][1]))

    for prediction_id, actual in pending:
        if rng.random() >= feedback_ratio:
            continue
        response = client.post("/feedback", json={"prediction_id": prediction_id, "actual": actual})
        if response.status_code == 201:
            report.feedbacks += 1
            report.absolute_errors.append(response.json()["absolute_error"])
    return report


def default_source() -> Path:
    """Lot le plus récent parmi les lots validés, sinon parmi tous les lots."""
    paths = get_settings().paths
    batches = list_batches(paths.processed_dir) or list_batches(paths.batches_dir)
    if not batches:
        raise SystemExit("Aucun lot disponible : lancer d'abord `make data`.")
    return batches[-1]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--count", type=int, default=200, help="nombre d'heures à prédire")
    parser.add_argument("--source", type=Path, help="lot CSV (défaut : dernier lot validé)")
    parser.add_argument("--feedback-ratio", type=float, default=0.8)
    parser.add_argument("--invalid-ratio", type=float, default=0.03)
    parser.add_argument("--batch-size", type=int, default=0, help="0 = prédictions unitaires")
    args = parser.parse_args(argv)

    source = args.source or default_source()
    hours = sample_hours(source, args.count)
    with httpx.Client(base_url=args.url, timeout=30) as client:
        report = send_traffic(
            client, hours, args.feedback_ratio, args.invalid_ratio, args.batch_size
        )
    print(f"{len(hours)} heures de {source.name} : {report.summary()}")


if __name__ == "__main__":
    main()
