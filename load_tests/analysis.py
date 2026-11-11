"""Analyse des tests de charge : mesures, critère de service, recommandation, rapport."""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

# Couleurs : palette catégorielle en ordre fixe (un worker = une couleur, sur tous les graphiques).
WORKER_COLORS = {1: "#2a78d6", 2: "#eb6834", 4: "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


@dataclass(frozen=True)
class Profile:
    """Limites de ressources du conteneur de l'API."""

    name: str
    cpus: float
    memory: str

    @property
    def label(self) -> str:
        return f"{self.cpus:g} vCPU / {self.memory.replace('m', ' Mo').replace('g', ' Go')}"


@dataclass(frozen=True)
class Slo:
    """Objectif de service : latence p95 de /predict et taux d'erreur maximaux."""

    p95_ms: float = 200.0
    max_error_rate: float = 0.01


@dataclass
class RunResult:
    profile: str
    cpus: float
    memory: str
    workers: int
    users: int
    status: str
    requests: int = 0
    failures: int = 0
    rps: float = 0.0
    p50_ms: float = math.nan
    p95_ms: float = math.nan
    p99_ms: float = math.nan
    predict_p95_ms: float = math.nan
    cpu_avg_pct: float = math.nan
    cpu_max_pct: float = math.nan
    mem_max_mb: float = math.nan

    @property
    def error_rate(self) -> float:
        return self.failures / self.requests if self.requests else 1.0

    def meets(self, slo: Slo) -> bool:
        return (
            self.status == "ok"
            and self.error_rate < slo.max_error_rate
            and self.predict_p95_ms < slo.p95_ms
        )


def _number(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return math.nan


def parse_locust_stats(path: Path) -> dict[str, float]:
    """Ligne « Aggregated » et ligne « /predict » du fichier ``*_stats.csv`` de Locust."""
    with path.open(encoding="utf-8") as handle:
        rows = {row["Name"]: row for row in csv.DictReader(handle)}
    total, predict = rows["Aggregated"], rows.get("/predict", {})
    return {
        "requests": int(total["Request Count"]),
        "failures": int(total["Failure Count"]),
        "rps": _number(total["Requests/s"]),
        "p50_ms": _number(total["50%"]),
        "p95_ms": _number(total["95%"]),
        "p99_ms": _number(total["99%"]),
        "predict_p95_ms": _number(predict.get("95%", "nan")),
    }


def _megabytes(text: str) -> float:
    # docker stats affiche des unités binaires (MiB) ; les limites docker (« 512m », « 1g ») aussi.
    units = {
        "B": 1e-6,
        "KIB": 1 / 1024,
        "MIB": 1.0,
        "GIB": 1024.0,
        "KB": 1e-3,
        "MB": 1.0,
        "GB": 1024.0,
    }
    value = text.strip().upper()
    for unit in sorted(units, key=len, reverse=True):
        if value.endswith(unit):
            return float(value[: -len(unit)]) * units[unit]
    return math.nan


def parse_docker_stats(lines: list[str]) -> dict[str, float]:
    """Lignes ``CPU%;MemUsage`` (format de ``docker stats``) → CPU moyen/max, mémoire max."""
    cpus, memories = [], []
    for line in lines:
        cpu, _, memory = line.partition(";")
        if not cpu.strip().endswith("%"):
            continue
        cpus.append(float(cpu.strip().rstrip("%")))
        memories.append(_megabytes(memory.split("/")[0]))
    if not cpus:
        return {"cpu_avg_pct": math.nan, "cpu_max_pct": math.nan, "mem_max_mb": math.nan}
    return {
        "cpu_avg_pct": sum(cpus) / len(cpus),
        "cpu_max_pct": max(cpus),
        "mem_max_mb": max(memories),
    }


def _size(result: RunResult) -> tuple[float, float, int]:
    return result.cpus, _megabytes(result.memory + "B"), result.workers


def capacity(results: list[RunResult], slo: Slo) -> dict[tuple[str, int], RunResult | None]:
    """Pour chaque configuration (profil, workers) : mesure au plus fort débit tenant le SLO."""
    table: dict[tuple[str, int], RunResult | None] = {}
    for result in results:
        key = (result.profile, result.workers)
        best = table.get(key)
        if result.meets(slo) and (best is None or result.rps > best.rps):
            table[key] = result
        else:
            table.setdefault(key, None)
    return table


def recommend(results: list[RunResult], slo: Slo, target_rps: float) -> str:
    """Plus petite configuration tenant le SLO à la plus forte charge où c'est possible,
    et nombre d'instances de cette taille pour atteindre ``target_rps``."""
    levels = sorted({result.users for result in results}, reverse=True)
    for users in levels:
        passing = [r for r in results if r.users == users and r.meets(slo)]
        if passing:
            break
    else:
        return "Aucune configuration testée ne respecte le SLO, même à la plus faible charge."

    chosen = min(passing, key=_size)
    text = (
        f"**{chosen.workers} worker(s) uvicorn, {chosen.cpus:g} vCPU, {chosen.memory} par "
        f"instance** : {chosen.rps:.0f} requêtes/s avec {users} utilisateurs simultanés, "
        f"p95 /predict = {chosen.predict_p95_ms:.0f} ms (objectif < {slo.p95_ms:.0f} ms), "
        f"{chosen.error_rate:.2%} d'erreurs, CPU moyen {chosen.cpu_avg_pct:.0f} %, "
        f"mémoire max {chosen.mem_max_mb:.0f} Mo."
    )
    if users < levels[0]:
        text += (
            f" À {levels[0]} utilisateurs, aucune configuration testée ne tient le SLO : la "
            "latence explose dès que le CPU sature, il faut répartir la charge sur plusieurs "
            "instances."
        )
    instances = math.ceil(target_rps / chosen.rps)
    text += (
        f" Pour un besoin de {target_rps:.0f} requêtes/s, prévoir **{instances} instance(s)** de "
        "cette taille derrière un répartiteur de charge (dimensionnement linéaire, à vérifier "
        "par un test sur plusieurs instances)."
    )
    return text


def capacity_table(results: list[RunResult], slo: Slo) -> str:
    rows = [
        "| Ressources | Workers | Débit max sous SLO | à N utilisateurs | p95 /predict |",
        "|---|---|---|---|---|",
    ]
    by_key = {(r.profile, r.workers): r for r in results}
    for (profile, workers), best in capacity(results, slo).items():
        reference = by_key[(profile, workers)]
        size = f"{reference.cpus:g} vCPU / {reference.memory}"
        if best is None:
            rows.append(f"| {size} | {workers} | SLO jamais tenu | - | - |")
        else:
            rows.append(
                f"| {size} | {workers} | {best.rps:.1f} req/s | {best.users} "
                f"| {best.predict_p95_ms:.0f} ms |"
            )
    return "\n".join(rows) + "\n"


def load_results(path: Path) -> list[RunResult]:
    """Relit un ``results.csv`` produit par ``write_report`` (pour régénérer le rapport)."""
    numeric = {f.name: f.type for f in fields(RunResult)}
    results = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            values: dict[str, object] = {}
            for name, raw in row.items():
                kind = numeric[name]
                values[name] = (
                    int(raw) if kind == "int" else _number(raw) if kind == "float" else raw
                )
            results.append(RunResult(**values))  # type: ignore[arg-type]
    return results


def _fmt(value: float, pattern: str) -> str:
    return "-" if isinstance(value, float) and math.isnan(value) else format(value, pattern)


def markdown_table(results: list[RunResult], slo: Slo) -> str:
    header = (
        "| Ressources | Workers | Utilisateurs | Statut | req/s | p50 (ms) | p95 (ms) | "
        "p99 (ms) | p95 /predict (ms) | Erreurs | CPU moy. | CPU max | RAM max (Mo) | SLO |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for r in results:
        lines.append(
            f"| {r.cpus:g} vCPU / {r.memory} | {r.workers} | {r.users} | {r.status} "
            f"| {_fmt(r.rps, '.1f')} | {_fmt(r.p50_ms, '.0f')} | {_fmt(r.p95_ms, '.0f')} "
            f"| {_fmt(r.p99_ms, '.0f')} | {_fmt(r.predict_p95_ms, '.0f')} "
            f"| {r.error_rate:.1%} | {_fmt(r.cpu_avg_pct, '.0f')} % "
            f"| {_fmt(r.cpu_max_pct, '.0f')} % "
            f"| {_fmt(r.mem_max_mb, '.0f')} | {'oui' if r.meets(slo) else 'non'} |"
        )
    return header + "\n".join(lines) + "\n"


def _small_multiples(
    results: list[RunResult],
    profiles: list[Profile],
    metric: str,
    ylabel: str,
    path: Path,
    threshold: float | None = None,
    log_scale: bool = False,
) -> None:
    """Un panneau par profil de ressources, une ligne par nombre de workers (axe y commun)."""
    figure, grid = plt.subplots(
        1, len(profiles), figsize=(4.2 * len(profiles), 3.6), sharey=True, squeeze=False
    )
    axes = list(grid[0])
    for axis, profile in zip(axes, profiles, strict=True):
        for workers, color in WORKER_COLORS.items():
            points = sorted(
                (r.users, getattr(r, metric))
                for r in results
                if r.profile == profile.name and r.workers == workers and r.status == "ok"
            )
            if points:
                users, values = zip(*points, strict=True)
                axis.plot(
                    users,
                    values,
                    color=color,
                    linewidth=2,
                    marker="o",
                    markersize=5,
                    label=f"{workers} worker{'s' if workers > 1 else ''}",
                )
        if threshold is not None:
            axis.axhline(threshold, color=MUTED, linewidth=1, linestyle="--")
            axis.annotate(
                "SLO",
                (1.0, threshold),
                xycoords=("axes fraction", "data"),
                ha="right",
                va="bottom",
                color=MUTED,
                fontsize=8,
            )
        if log_scale:
            axis.set_yscale("log")
        axis.set_title(profile.label, color=INK, fontsize=10)
        axis.set_xlabel("utilisateurs simultanés", color=MUTED)
        axis.grid(color=GRID, linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(ylabel, color=MUTED)
    axes[0].legend(frameon=False, fontsize=9)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def write_report(
    results: list[RunResult],
    profiles: list[Profile],
    slo: Slo,
    out_dir: Path,
    protocol: str,
    target_rps: float,
) -> Path:
    """Écrit results.csv, trois graphiques et RESULTATS.md (recommandation, capacité, mesures)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[f.name for f in fields(RunResult)])
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)

    _small_multiples(
        results,
        profiles,
        "predict_p95_ms",
        "p95 /predict (ms, échelle log)",
        out_dir / "latence_p95.png",
        threshold=slo.p95_ms,
        log_scale=True,
    )
    _small_multiples(results, profiles, "rps", "requêtes / s", out_dir / "debit.png")
    _small_multiples(results, profiles, "cpu_avg_pct", "CPU moyen (%)", out_dir / "cpu.png")

    report = out_dir / "RESULTATS.md"
    report.write_text(
        "# Résultats des tests de charge\n\n"
        f"{protocol}\n\n"
        f"## Recommandation\n\n{recommend(results, slo, target_rps)}\n\n"
        "## Capacité de chaque configuration sous SLO\n\n"
        f"{capacity_table(results, slo)}\n"
        "## Mesures\n\n"
        f"{markdown_table(results, slo)}\n"
        "## Graphiques\n\n"
        "![Latence p95 de /predict](latence_p95.png)\n\n"
        "![Débit](debit.png)\n\n"
        "![CPU](cpu.png)\n",
        encoding="utf-8",
    )
    return report
