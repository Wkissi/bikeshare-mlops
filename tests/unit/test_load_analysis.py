"""Analyse des tests de charge : lecture des mesures, SLO, recommandation, rapport."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from load_tests.analysis import (
    Profile,
    RunResult,
    Slo,
    capacity,
    capacity_table,
    load_results,
    parse_docker_stats,
    parse_locust_stats,
    recommend,
    write_report,
)

LOCUST_CSV = (
    '"Type","Name","Request Count","Failure Count","Median Response Time",'
    '"Average Response Time","Min Response Time","Max Response Time","Average Content Size",'
    '"Requests/s","Failures/s","50%","66%","75%","80%","90%","95%","98%","99%","99.9%",'
    '"99.99%","100%"\n'
    '"POST","/predict","800","2","30","35","10","300","150","20.0","0.05","30","35","40","45",'
    '"60","80","110","140","290","300","300"\n'
    '"POST","/feedback","100","0","20","22","8","100","120","2.5","0","20","22","25","27","30",'
    '"40","60","70","100","100","100"\n'
    '"","Aggregated","900","2","28","33","8","300","145","22.5","0.05","28","33","38","43",'
    '"58","78","105","135","290","300","300"\n'
)

SMALL, MEDIUM = Profile("small", 1, "512m"), Profile("medium", 2, "1g")


def _result(profile: Profile, workers: int, users: int, p95: float, **extra: float) -> RunResult:
    values = {"requests": 1000, "failures": 0, "rps": 50.0, "cpu_avg_pct": 40.0, "mem_max_mb": 400}
    values |= extra
    return RunResult(
        profile.name, profile.cpus, profile.memory, workers, users, "ok",
        predict_p95_ms=p95, p50_ms=20, p95_ms=p95, p99_ms=p95 * 1.5, cpu_max_pct=80, **values,
    )  # fmt: skip


def test_parse_locust_stats(tmp_path: Path) -> None:
    path = tmp_path / "run_stats.csv"
    path.write_text(LOCUST_CSV, encoding="utf-8")

    stats = parse_locust_stats(path)

    assert stats["requests"] == 900
    assert stats["failures"] == 2
    assert stats["rps"] == 22.5
    assert stats["p95_ms"] == 78
    assert stats["predict_p95_ms"] == 80


def test_parse_docker_stats() -> None:
    lines = ["12.5%;300MiB / 1GiB", "47.5%;1.5GiB / 2GiB", "--;", ""]

    stats = parse_docker_stats(lines)

    assert stats["cpu_avg_pct"] == 30.0
    assert stats["cpu_max_pct"] == 47.5
    assert stats["mem_max_mb"] == 1536.0
    assert math.isnan(parse_docker_stats([])["cpu_avg_pct"])


def test_slo_requires_latency_errors_and_running_api() -> None:
    slo = Slo(p95_ms=200, max_error_rate=0.01)
    assert _result(SMALL, 1, 100, 150).meets(slo)
    assert not _result(SMALL, 1, 100, 250).meets(slo)
    assert not _result(SMALL, 1, 100, 150, failures=20).meets(slo)
    down = RunResult("small", 1, "512m", 4, 100, "api_indisponible")
    assert not down.meets(slo) and down.error_rate == 1.0


def test_recommendation_picks_smallest_passing_config_at_peak() -> None:
    results = [
        _result(SMALL, 1, 100, 450),
        _result(MEDIUM, 2, 100, 120, rps=40),
        _result(MEDIUM, 4, 100, 90),
        _result(SMALL, 2, 10, 50),  # passe, mais pas à la charge maximale
    ]

    text = recommend(results, Slo(), target_rps=100)

    assert "2 worker(s) uvicorn, 2 vCPU, 1g" in text
    assert "40 requêtes/s avec 100 utilisateurs" in text
    assert "**3 instance(s)**" in text
    assert "aucune configuration testée ne tient" not in text


def test_recommendation_falls_back_to_highest_load_with_a_passing_config() -> None:
    results = [
        _result(SMALL, 1, 100, 4000),
        _result(MEDIUM, 2, 100, 1500),
        _result(SMALL, 2, 50, 180, rps=30),
        _result(MEDIUM, 4, 50, 150, rps=45),
        _result(SMALL, 1, 10, 90, rps=12),
    ]

    text = recommend(results, Slo(), target_rps=100)

    assert "2 worker(s) uvicorn, 1 vCPU, 512m" in text
    assert "avec 50 utilisateurs" in text
    assert "À 100 utilisateurs, aucune configuration testée ne tient le SLO" in text
    assert "**4 instance(s)**" in text


def test_recommendation_when_nothing_passes() -> None:
    results = [
        _result(SMALL, 1, 100, 450),
        RunResult("small", 1, "512m", 4, 10, "api_indisponible"),
    ]
    assert "Aucune configuration" in recommend(results, Slo(), target_rps=100)


def test_capacity_keeps_best_throughput_under_slo() -> None:
    results = [
        _result(MEDIUM, 2, 10, 80, rps=12),
        _result(MEDIUM, 2, 50, 150, rps=45),
        _result(MEDIUM, 2, 100, 900, rps=50),
        _result(SMALL, 1, 10, 300, rps=10),
    ]

    table = capacity(results, Slo())

    assert table[("medium", 2)] is not None and table[("medium", 2)].rps == 45
    assert table[("small", 1)] is None
    assert "SLO jamais tenu" in capacity_table(results, Slo())


def test_results_roundtrip_through_csv(tmp_path: Path) -> None:
    results = [_result(MEDIUM, 2, 10, 80), RunResult("small", 1, "512m", 4, 10, "api_indisponible")]
    write_report(results, [SMALL, MEDIUM], Slo(), tmp_path, "P.", target_rps=100)

    loaded = load_results(tmp_path / "results.csv")

    assert loaded[0] == results[0]
    assert loaded[1].status == "api_indisponible" and math.isnan(loaded[1].rps) is False


@pytest.mark.parametrize("profiles", [[MEDIUM], [SMALL, MEDIUM]])
def test_report_is_written(tmp_path: Path, profiles: list[Profile]) -> None:
    results = [_result(p, w, u, 50 + u) for p in profiles for w in (1, 2) for u in (10, 100)]

    report = write_report(results, profiles, Slo(), tmp_path, "Protocole de test.", 50)

    content = report.read_text(encoding="utf-8")
    assert "## Recommandation" in content and "Protocole de test." in content
    assert content.count("| oui |") == len(results)
    for name in ("results.csv", "latence_p95.png", "debit.png", "cpu.png"):
        assert (tmp_path / name).stat().st_size > 0
