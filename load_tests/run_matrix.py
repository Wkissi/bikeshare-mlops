"""Exécute la matrice de tests de charge sur la stack Docker et produit le rapport.

Pour chaque profil de ressources (vCPU, mémoire) et chaque nombre de workers uvicorn :
1. recrée le conteneur ``api`` avec ces limites (variables API_CPUS, API_MEMORY, API_WORKERS) ;
2. pour chaque niveau d'utilisateurs : échauffement, puis Locust en mode headless pendant
   ``--duration`` secondes (statistiques remises à zéro après la montée en charge), avec relevé
   ``docker stats`` toutes les 2 s ;
3. rassemble les mesures dans ``docs/load_tests/`` (CSV, graphiques, RESULTATS.md).

Usage : uv run python -m load_tests.run_matrix
        [--profiles medium] [--workers 2] [--users 10 20 30] [--append]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

from load_tests.analysis import (
    Profile,
    RunResult,
    Slo,
    load_results,
    parse_docker_stats,
    parse_locust_stats,
    write_report,
)

API = "http://localhost:8000"
CONTAINER = "bikeshare-api-1"
RAW_DIR = Path("reports/load_tests")
OUT_DIR = Path("docs/load_tests")
PROFILES = [
    Profile("small", 1, "512m"),
    Profile("medium", 2, "1g"),
    Profile("large", 4, "2g"),
]
WORKERS = [1, 2, 4]
USERS = [10, 50, 100]
# Besoin cible retenu pour le dimensionnement : 100 utilisateurs avec 0,5 à 1 s de réflexion
# offrent environ 100 requêtes/s.
TARGET_RPS = 100.0


def restart_api(profile: Profile, workers: int) -> bool:
    """Recrée l'API avec ces limites ; False si elle ne devient pas disponible (ex. OOM)."""
    env = os.environ | {
        "API_CPUS": str(profile.cpus),
        "API_MEMORY": profile.memory,
        "API_WORKERS": str(workers),
    }
    subprocess.run(
        ["docker", "compose", "up", "-d", "--no-deps", "--force-recreate", "api"],
        env=env,
        check=True,
        capture_output=True,
    )
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{API}/health", timeout=5).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(3)
    return False


def warm_up(requests: int = 30) -> None:
    from bikeshare_mlops.monitoring.simulate import default_source, sample_hours

    with httpx.Client(base_url=API, timeout=30) as client:
        for features, _ in sample_hours(default_source(), requests, seed=7):
            client.post("/predict", json=features)


class DockerStatsSampler(threading.Thread):
    """Relève CPU et mémoire du conteneur de l'API pendant un test."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.lines: list[str] = []
        self._halt = threading.Event()

    def run(self) -> None:
        while not self._halt.is_set():
            result = subprocess.run(
                [
                    "docker",
                    "stats",
                    "--no-stream",
                    "--format",
                    "{{.CPUPerc}};{{.MemUsage}}",
                    CONTAINER,
                ],
                capture_output=True,
                text=True,
            )
            self.lines.extend(result.stdout.splitlines())
            self._halt.wait(2)

    def stop(self) -> list[str]:
        self._halt.set()
        self.join()
        return self.lines


def run_locust(users: int, duration_s: int, prefix: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "locust",
            "-f",
            "load_tests/locustfile.py",
            "--headless",
            "--host",
            API,
            "-u",
            str(users),
            "-r",
            str(max(1, users // 5)),
            "-t",
            f"{duration_s}s",
            "--reset-stats",
            "--only-summary",
            "--csv",
            str(prefix),
        ],
        check=False,
        capture_output=True,
    )


def measure(profile: Profile, workers: int, users: int, duration_s: int) -> RunResult:
    prefix = RAW_DIR / f"{profile.name}_w{workers}_u{users}"
    sampler = DockerStatsSampler()
    sampler.start()
    run_locust(users, duration_s, prefix)
    stats = parse_docker_stats(sampler.stop())
    result = RunResult(profile.name, profile.cpus, profile.memory, workers, users, "ok")
    for key, value in (parse_locust_stats(Path(f"{prefix}_stats.csv")) | stats).items():
        setattr(result, key, value)
    return result


def protocol_text(duration_s: int) -> str:
    return (
        "Protocole : API seule limitée en CPU et mémoire (`docker compose`, deploy.resources), "
        "PostgreSQL et MLflow non limités, même machine. Locust headless depuis l'hôte, "
        f"{duration_s} s mesurées par test après la montée en charge, temps de réflexion de 0,5 à "
        "1 s par utilisateur ; 80 % /predict, 10 % /predict/batch (10 heures), 10 % /feedback. "
        "Objectif de service (SLO) : p95 de /predict < 200 ms et moins de 1 % d'erreurs. "
        f"Besoin cible pour le dimensionnement : {TARGET_RPS:.0f} requêtes/s."
    )


def _write(results: list[RunResult], duration_s: int) -> None:
    results = sorted(
        results, key=lambda r: ([p.name for p in PROFILES].index(r.profile), r.workers, r.users)
    )
    names = {result.profile for result in results}
    profiles = [profile for profile in PROFILES if profile.name in names]
    report = write_report(results, profiles, Slo(), OUT_DIR, protocol_text(duration_s), TARGET_RPS)
    print(f"Rapport : {report}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=int, default=40, help="secondes mesurées par test")
    parser.add_argument("--profiles", nargs="+", choices=[p.name for p in PROFILES])
    parser.add_argument("--workers", nargs="+", type=int, default=WORKERS)
    parser.add_argument("--users", nargs="+", type=int, default=USERS)
    parser.add_argument(
        "--append", action="store_true", help="complète results.csv au lieu de le remplacer"
    )
    parser.add_argument(
        "--report-only", action="store_true", help="régénère le rapport depuis results.csv"
    )
    args = parser.parse_args()
    previous = load_results(OUT_DIR / "results.csv") if args.append or args.report_only else []
    if args.report_only:
        _write(previous, args.duration)
        return

    profiles = [p for p in PROFILES if args.profiles is None or p.name in args.profiles]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    results: list[RunResult] = []
    for profile in profiles:
        for workers in args.workers:
            print(f"[{profile.label}, {workers} worker(s)] démarrage de l'API…", flush=True)
            if not restart_api(profile, workers):
                print("    API indisponible (mémoire insuffisante ?) : configuration écartée")
                results.extend(
                    RunResult(
                        profile.name,
                        profile.cpus,
                        profile.memory,
                        workers,
                        users,
                        "api_indisponible",
                    )
                    for users in args.users
                )
                continue
            warm_up()
            for users in args.users:
                result = measure(profile, workers, users, args.duration)
                results.append(result)
                print(
                    f"    {users:>3} utilisateurs : {result.rps:6.1f} req/s, "
                    f"p95 /predict {result.predict_p95_ms:6.0f} ms, "
                    f"erreurs {result.error_rate:.1%}, CPU {result.cpu_avg_pct:.0f} %, "
                    f"RAM {result.mem_max_mb:.0f} Mo",
                    flush=True,
                )

    measured = {(r.profile, r.workers, r.users) for r in results}
    kept = [r for r in previous if (r.profile, r.workers, r.users) not in measured]
    _write(kept + results, args.duration)
    restart_api(Profile("default", 2, "1g"), 2)


if __name__ == "__main__":
    main()
