"""Générateur de données synthétiques au format ``hour.csv`` (aucun téléchargement en test)."""

from __future__ import annotations

import numpy as np
import pandas as pd

HOURLY_PROFILE = np.array(
    [0.2, 0.1, 0.08, 0.05, 0.05, 0.15, 0.5, 1.2, 2.0, 1.1, 0.8, 0.9,
     1.1, 1.1, 1.0, 1.1, 1.5, 2.2, 2.0, 1.4, 1.0, 0.8, 0.6, 0.4]
)  # fmt: skip


def make_hourly_frame(start: str = "2011-01-01", days: int = 31, seed: int = 0) -> pd.DataFrame:
    """Jeu horaire réaliste et cohérent (saisons, jours ouvrés, cnt = casual + registered)."""
    rng = np.random.default_rng(seed)
    stamps = pd.date_range(start, periods=days * 24, freq="h")
    month = stamps.month.to_numpy()
    weekday = ((stamps.dayofweek + 1) % 7).to_numpy()  # convention UCI : 0 = dimanche
    workingday = ((weekday >= 1) & (weekday <= 5)).astype(int)
    seasonal = 0.5 + 0.3 * np.sin((month - 4) / 12 * 2 * np.pi)
    temp = np.clip(seasonal + rng.normal(0, 0.05, len(stamps)), 0, 1)
    weathersit = rng.choice([1, 2, 3], size=len(stamps), p=[0.65, 0.27, 0.08])
    base = HOURLY_PROFILE[stamps.hour] * (40 + 200 * temp) * (1.2 - 0.25 * (weathersit - 1))
    registered = rng.poisson(base * (0.6 + 0.3 * workingday)).astype(int)
    casual = rng.poisson(base * (0.4 - 0.25 * workingday)).astype(int)
    frame = pd.DataFrame(
        {
            "instant": np.arange(1, len(stamps) + 1),
            "dteday": stamps.strftime("%Y-%m-%d"),
            "season": ((month % 12) // 3 + 1),
            "yr": stamps.year.to_numpy() - 2011,
            "mnth": month,
            "hr": stamps.hour.to_numpy(),
            "holiday": 0,
            "weekday": weekday,
            "workingday": workingday,
            "weathersit": weathersit,
            "temp": temp.round(2),
            "atemp": np.clip(temp * 0.95, 0, 1).round(4),
            "hum": rng.uniform(0.2, 1.0, len(stamps)).round(2),
            "windspeed": rng.uniform(0.0, 0.6, len(stamps)).round(4),
            "casual": casual,
            "registered": registered,
            "cnt": casual + registered,
        }
    )
    # Mêmes types qu'un CSV relu par pandas (sous Windows, numpy produit parfois de l'int32).
    integer_columns = frame.select_dtypes("integer").columns
    return frame.astype(dict.fromkeys(integer_columns, "int64"))
