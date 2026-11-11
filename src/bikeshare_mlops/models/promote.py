"""Règles de mise en service : un challenger ne remplace le champion que s'il le mérite.

Conditions (seuils dans ``config.yaml`` > ``promotion``), évaluées sur le MÊME jeu de test :
1. tests de non-régression, tous obligatoires :
   - prédictions finies et positives ;
   - RMSE nettement sous la baseline naïve ;
   - mêmes colonnes d'entrée que le champion (l'API n'a pas à changer) ;
   - aucune saison dégradée au-delà du seuil par rapport au champion ;
   - latence d'une prédiction unitaire sous le seuil ;
2. gain de RMSE d'au moins ``min_improvement_pct`` % sur le champion actuel.
Sans champion (premier modèle), seuls les tests de non-régression s'appliquent.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from sklearn.pipeline import Pipeline

from bikeshare_mlops.config import Settings, get_settings
from bikeshare_mlops.data.dataset import load_batches, train_test_split_stable
from bikeshare_mlops.exceptions import RegistryError
from bikeshare_mlops.features.build import TARGET, split_features_target
from bikeshare_mlops.models import registry
from bikeshare_mlops.models.baseline import HourlyMeanBaseline
from bikeshare_mlops.models.evaluate import compute_metrics, metrics_by_segment
from bikeshare_mlops.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

_LATENCY_SAMPLES = 20


@dataclass(frozen=True)
class CheckResult:
    """Résultat d'un test de non-régression."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class PromotionDecision:
    """Décision motivée, enregistrée en tags sur la version du registre."""

    version: str
    promoted: bool
    reason: str
    challenger_rmse: float
    champion_version: str | None
    champion_rmse: float | None
    improvement_pct: float | None
    checks: tuple[CheckResult, ...]

    def as_tags(self) -> dict[str, str]:
        def _fmt(value: float | None, digits: int) -> str:
            return "-" if value is None else f"{value:.{digits}f}"

        return {
            registry.TAG_STATUS: "promoted" if self.promoted else "rejected",
            registry.TAG_REASON: self.reason,
            "eval_rmse": f"{self.challenger_rmse:.4f}",
            "eval_champion_version": self.champion_version or "-",
            "eval_champion_rmse": _fmt(self.champion_rmse, 4),
            "eval_improvement_pct": _fmt(self.improvement_pct, 2),
            "eval_checks": json.dumps([asdict(check) for check in self.checks], ensure_ascii=False),
            "decided_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }


def _median_single_prediction_ms(model: Pipeline, row: pd.DataFrame) -> float:
    durations = []
    for _ in range(_LATENCY_SAMPLES):
        start = time.perf_counter()
        model.predict(row)
        durations.append((time.perf_counter() - start) * 1000)
    return statistics.median(durations)


def non_regression_checks(
    challenger: Pipeline,
    champion: Pipeline | None,
    frame: pd.DataFrame,
    settings: Settings,
) -> tuple[list[CheckResult], float, float | None]:
    """Exécute les tests de non-régression ; renvoie aussi les RMSE challenger et champion."""
    rules = settings.promotion
    train, test = train_test_split_stable(
        frame, settings.training.test_fraction, settings.training.split_salt
    )
    x_train, y_train = split_features_target(train)
    x_test, y_test = split_features_target(test)

    y_challenger = challenger.predict(x_test)
    challenger_rmse = compute_metrics(y_test, y_challenger).rmse
    baseline_rmse = compute_metrics(
        y_test, HourlyMeanBaseline().fit(x_train, y_train).predict(x_test)
    ).rmse
    max_rmse = baseline_rmse * (1 - rules.min_gain_vs_baseline_pct / 100)

    checks = [
        CheckResult(
            "predictions_valid",
            bool(np.isfinite(y_challenger).all() and (y_challenger >= 0).all()),
            "prédictions finies et >= 0",
        ),
        CheckResult(
            "beats_baseline",
            challenger_rmse <= max_rmse,
            f"rmse {challenger_rmse:.2f} <= {max_rmse:.2f} (baseline {baseline_rmse:.2f})",
        ),
    ]
    latency = _median_single_prediction_ms(challenger, x_test.head(1))
    checks.append(
        CheckResult(
            "single_prediction_latency",
            latency <= rules.max_single_prediction_ms,
            f"médiane {latency:.1f} ms <= {rules.max_single_prediction_ms:.0f} ms",
        )
    )

    champion_rmse = None
    if champion is not None:
        y_champion = champion.predict(x_test)
        champion_rmse = compute_metrics(y_test, y_champion).rmse
        same_inputs = list(challenger.feature_names_in_) == list(champion.feature_names_in_)
        checks.append(CheckResult("same_input_schema", same_inputs, "colonnes d'entrée identiques"))
        challenger_segments = metrics_by_segment(test, y_challenger, TARGET)
        champion_segments = metrics_by_segment(test, y_champion, TARGET)
        tolerance = 1 + rules.max_segment_degradation_pct / 100
        degraded = [
            name
            for name, metrics in challenger_segments.items()
            if metrics.rmse > champion_segments[name].rmse * tolerance
        ]
        checks.append(
            CheckResult(
                "segments_not_degraded",
                not degraded,
                f"saisons dégradées > {rules.max_segment_degradation_pct:.0f} % : "
                f"{degraded or 'aucune'}",
            )
        )
    return checks, challenger_rmse, champion_rmse


def decide(
    version: str,
    challenger: Pipeline,
    champion: Pipeline | None,
    champion_version: str | None,
    frame: pd.DataFrame,
    settings: Settings,
) -> PromotionDecision:
    """Décision de promotion, sans effet de bord (testable sans MLflow)."""
    checks, challenger_rmse, champion_rmse = non_regression_checks(
        challenger, champion, frame, settings
    )
    failed = [check.name for check in checks if not check.passed]
    improvement = None
    if champion_rmse is not None:
        improvement = (champion_rmse - challenger_rmse) / champion_rmse * 100

    if failed:
        promoted, reason = False, f"non-régression échouée : {', '.join(failed)}"
    elif improvement is None:
        promoted, reason = True, "premier modèle : tests de non-régression validés"
    elif improvement >= settings.promotion.min_improvement_pct:
        promoted, reason = True, f"gain de {improvement:.2f} % sur le champion v{champion_version}"
    else:
        promoted = False
        reason = (
            f"gain insuffisant : {improvement:.2f} % < "
            f"{settings.promotion.min_improvement_pct:.2f} % requis"
        )
    return PromotionDecision(
        version=version,
        promoted=promoted,
        reason=reason,
        challenger_rmse=challenger_rmse,
        champion_version=champion_version,
        champion_rmse=champion_rmse,
        improvement_pct=improvement,
        checks=tuple(checks),
    )


def _load(settings: Settings, version: str) -> Pipeline:
    uri = f"models:/{settings.mlflow.registered_model_name}/{version}"
    return mlflow.sklearn.load_model(uri)


def evaluate_and_promote(
    settings: Settings, frame: pd.DataFrame, version: str | None = None
) -> PromotionDecision:
    """Compare la candidate (défaut : alias challenger) au champion et applique la décision."""
    registry.configure_tracking(settings)
    challenger_version = registry.version_by_alias(settings, registry.CHALLENGER)
    version = version or challenger_version
    if version is None:
        raise RegistryError("Aucun challenger à évaluer : lancer d'abord un entraînement.")

    champion_version = registry.version_by_alias(settings, registry.CHAMPION)
    if champion_version == version:
        raise RegistryError(f"La version {version} est déjà le champion.")
    champion = None if champion_version is None else _load(settings, champion_version)

    decision = decide(
        version, _load(settings, version), champion, champion_version, frame, settings
    )
    registry.tag_version(settings, version, decision.as_tags())
    if decision.promoted:
        registry.set_alias(settings, registry.CHAMPION, version)
        if challenger_version == version:
            registry.delete_alias(settings, registry.CHALLENGER)
        if champion_version is not None:
            registry.tag_version(settings, champion_version, {"replaced_by": version})
    log = logger.info if decision.promoted else logger.warning
    log(
        "décision de promotion",
        extra={
            "version": version,
            "promoted": decision.promoted,
            "reason": decision.reason,
            "champion_version": champion_version,
        },
    )
    return decision


def rollback(settings: Settings, version: str) -> str | None:
    """Remet l'alias champion sur ``version`` ; renvoie l'ancien champion."""
    registry.configure_tracking(settings)
    name = settings.mlflow.registered_model_name
    try:
        registry.client(settings).get_model_version(name, version)
    except MlflowException as error:
        raise RegistryError(f"La version {version} du modèle {name} n'existe pas.") from error

    previous = registry.version_by_alias(settings, registry.CHAMPION)
    if previous == version:
        raise RegistryError(f"La version {version} est déjà le champion.")

    registry.set_alias(settings, registry.CHAMPION, version)
    if registry.version_by_alias(settings, registry.CHALLENGER) == version:
        registry.delete_alias(settings, registry.CHALLENGER)
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    registry.tag_version(
        settings,
        version,
        {registry.TAG_STATUS: "promoted", "rollback_from": previous or "-", "restored_at": stamp},
    )
    if previous is not None:
        registry.tag_version(
            settings, previous, {registry.TAG_STATUS: "rolled_back", "rolled_back_at": stamp}
        )
    logger.warning("rollback du champion", extra={"from_version": previous, "to_version": version})
    return previous


def main(argv: list[str] | None = None) -> None:
    """``decide`` : évalue le challenger ; ``rollback --version n`` : restaure une version."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    decide_cmd = commands.add_parser("decide", help="évalue et promeut (ou rejette) un challenger")
    decide_cmd.add_argument("--data-dir", type=Path, default=settings.paths.processed_dir)
    decide_cmd.add_argument("--version", help="version à évaluer (défaut : alias challenger)")
    rollback_cmd = commands.add_parser("rollback", help="remet le champion sur une version")
    rollback_cmd.add_argument("--version", required=True)
    args = parser.parse_args(argv)

    configure_logging(settings.logging.level, settings.logging.json_format)
    try:
        if args.command == "decide":
            decision = evaluate_and_promote(settings, load_batches(args.data_dir), args.version)
            verdict = "PROMU" if decision.promoted else "REJETÉ"
            print(f"Version {decision.version} : {verdict} ({decision.reason})")
            for check in decision.checks:
                print(f"  [{'ok' if check.passed else 'KO'}] {check.name} : {check.detail}")
        else:
            previous = rollback(settings, args.version)
            print(f"Champion : v{previous or '-'} -> v{args.version}")
    except RegistryError as error:
        raise SystemExit(f"Erreur : {error}") from error


if __name__ == "__main__":
    main()
