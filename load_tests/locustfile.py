"""Scénario de charge : un client de l'API de prévision.

Mélange de requêtes (poids) : 80 % de prédictions unitaires, 10 % de prédictions par lot de
10 heures, 10 % de feedbacks sur des prédictions déjà obtenues. Temps de réflexion de 0,5 à 1 s
entre deux requêtes : 100 utilisateurs offrent environ 130 requêtes/s.

Les heures envoyées sont tirées d'un lot réel (LOAD_TEST_SOURCE, défaut : dernier lot validé).
"""

from __future__ import annotations

import os
import random
from pathlib import Path

from locust import HttpUser, between, events, task

from bikeshare_mlops.monitoring.simulate import default_source, sample_hours

HOURS: list[tuple[dict, float]] = []


@events.init.add_listener
def _load_hours(environment: object, **kwargs: object) -> None:
    source = Path(os.environ["LOAD_TEST_SOURCE"]) if "LOAD_TEST_SOURCE" in os.environ else None
    HOURS.extend(sample_hours(source or default_source(), 500, seed=42))


class ForecastClient(HttpUser):
    wait_time = between(0.5, 1.0)

    def on_start(self) -> None:
        self.predictions: list[tuple[str, float]] = []

    @task(8)
    def predict(self) -> None:
        features, actual = random.choice(HOURS)
        with self.client.post("/predict", json=features, catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
                return
            self.predictions.append((response.json()["prediction_id"], actual))
            del self.predictions[:-50]

    @task(1)
    def predict_batch(self) -> None:
        instances = [features for features, _ in random.sample(HOURS, 10)]
        with self.client.post(
            "/predict/batch", json={"instances": instances}, catch_response=True
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def feedback(self) -> None:
        if not self.predictions:
            return
        prediction_id, actual = self.predictions.pop()
        payload = {"prediction_id": prediction_id, "actual": actual}
        with self.client.post("/feedback", json=payload, catch_response=True) as response:
            if response.status_code != 201:
                response.failure(f"HTTP {response.status_code}")
