# Captures d'écran (preuves d'intégration)

Déposer ici les captures, sous ces noms exacts : `docs/SYNTHESE.md` les affiche déjà.
Préparation : `make up`, puis `make demo-nominal`, puis `make demo-incident HOLD=90` (la panne
PostgreSQL dure 90 s, le temps de faire les captures 07 à 09).

| Fichier | Quoi capturer | Où |
|---|---|---|
| `01-prefect-runs.png` | liste des runs : `watch_incoming` → `training_pipeline` réussis et le run en échec du lot invalide | http://localhost:4200 → *Runs* |
| `02-prefect-flow-graph.png` | graphe d'un run `training_pipeline` réussi (tâches ingest → promote) | clic sur un run → onglet graphe |
| `03-mlflow-run.png` | un run de `bikeshare-demand` : paramètres, métriques, tags `data_sha256` et `code_version` | http://localhost:5000 → *Experiments* |
| `04-mlflow-artifacts.png` | artefacts du même run : `evaluation/` (graphiques) et `model/` | onglet *Artifacts* |
| `05-mlflow-registry.png` | versions de `bikeshare-rf` avec les alias `champion` / `challenger` et les tags de promotion | *Models* → `bikeshare-rf` |
| `06-grafana-nominal.png` | dashboard après `make demo-nominal` : trafic, latences, prédictions par version, MAE | http://localhost:3000 |
| `07-grafana-incident.png` | dashboard pendant la panne : tuile « Base de données » rouge, pic de 503 | pendant `demo-incident` |
| `08-prometheus-alerts.png` | alertes `DatabaseDown` et `HighServerErrorRate` à l'état *firing* | http://localhost:9090/alerts |
| `09-api-health-degraded.png` | réponse de `GET /health` en 503 `degraded` | http://localhost:8000/docs |
| `10-alert-invalid-batch.png` | sortie de `make alerts` et rapport de validation du lot rejeté | terminal |
| `11-github-actions.png` | dernier run vert de la CI (lint + tests + couverture) | onglet *Actions* du dépôt |
