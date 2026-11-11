# bikeshare-mlops

Projet final 5MLDE (Supinfo) : industrialiser un modèle de prévision du nombre de locations
de vélos par heure (UCI Bike Sharing Dataset).

## Prérequis

- [uv](https://docs.astral.sh/uv/) (installe Python 3.11 et les dépendances figées)
- GNU Make
- Docker + Docker Compose

## Installation et vérification

```bash
make install   # Python 3.11 + dépendances figées dans uv.lock
make lint      # ruff
make test      # pytest + couverture
make help      # liste de toutes les commandes
```

## Données

- **Source :** [UCI Machine Learning Repository : Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)
  (Fanaee-T, H. & Gama, J., 2013), fichier `hour.csv` : 17 379 relevés horaires, 2011-2012,
  système Capital Bikeshare (Washington D.C.).
- **Licence :** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) : réutilisation libre
  avec citation de la source.
- **Récupération :** `make data` télécharge l'archive officielle, vérifie son SHA-256 (fixé dans
  `config/config.yaml`) et ne la retélécharge pas si elle est déjà présente et intègre. Les
  données ne sont jamais versionnées (`data/` est dans `.gitignore`).
- **Simulation de nouvelles données :** le dataset est découpé en 24 lots mensuels
  (`data/batches/batch_YYYY_MM.csv`). `make next-batch` fait « arriver » le lot suivant dans
  `data/incoming/`, comme le ferait un export mensuel du système de location.

| Commande | Effet |
|---|---|
| `make data` | téléchargement vérifié + découpage en lots |
| `make next-batch COUNT=n` | dépose les `n` prochains lots dans `data/incoming/` |
| `make validate FILES="..."` | contrôle qualité pandera (rapport JSON dans `reports/validation/`) |
| `make invalid-batch` | dépose un lot volontairement corrompu (démo d'incident) |

## Services Docker et cycle de vie des modèles

`make up` démarre PostgreSQL, MLflow et le serveur Prefect (images construites depuis `uv.lock`).

| Service | URL |
|---|---|
| MLflow (tracking + registre) | http://localhost:5000 |
| Prefect (orchestration) | http://localhost:4200 |
| PostgreSQL | `localhost:5432` (bases `bikeshare`, `mlflow`, `prefect`) |

| Commande | Effet |
|---|---|
| `make train DATA_DIR=data/processed` | entraîne, trace le run dans MLflow, enregistre la version en `challenger` |
| `make promote` | compare le challenger au champion sur le même jeu de test, promeut ou rejette (tag motivé) |
| `make models` | état du registre : versions, alias, statut |
| `make rollback VERSION=n` | remet l'alias `champion` sur la version `n` |
| `make train-local` | évaluation locale sans MLflow (`reports/training/`) |

Règles de promotion (`config/config.yaml` > `promotion`) : le challenger doit passer tous les tests de
non-régression (prédictions valides, RMSE au moins 30 % sous la baseline, mêmes colonnes d'entrée,
aucune saison dégradée de plus de 10 %, latence unitaire sous 200 ms) **et** battre le RMSE du
champion d'au moins 2 %.

## Orchestration (Prefect)

Le conteneur `pipeline-worker` sert trois déploiements (visibles sur http://localhost:4200) :

| Déploiement | Déclenchement | Rôle |
|---|---|---|
| `watch_incoming/watch-incoming` | toutes les 60 s | lance `training_pipeline` dès qu'un lot arrive dans `data/incoming/` |
| `training_pipeline/weekly-retrain` | cron `0 6 * * 1` | réentraîne sur toutes les données validées |
| `batch_prediction/on-demand` | manuel (UI ou CLI) | prédit un CSV avec le champion |

Flow `training_pipeline` : `ingest` → `validate` → `build_dataset` → `train` → `register` → `promote`.

- Chaque lot reçu donne un run MLflow dans l'expérience `bikeshare-ingestion` (hash, volume, période,
  rapport qualité).
- Un lot non conforme part dans `data/rejected/`, le flow échoue **avant** tout entraînement et une
  alerte est émise ; les lots valides du même passage restent en attente.
- Les tâches qui parlent à MLflow sont relancées 3 fois (10 s d'intervalle) en cas d'erreur.
- Alertes : `logs/alerts.jsonl` (toujours) + log Prefect + webhook Slack/Discord si
  `BIKESHARE_ALERTS__WEBHOOK_URL` est défini.

| Commande | Effet |
|---|---|
| `make next-batch` | dépose le lot suivant : le worker le traite dans la minute |
| `make invalid-batch` | dépose un lot corrompu : pipeline bloqué + alerte |
| `make pipeline` | exécute une fois le flow depuis l'hôte (`RETRAIN=1` sans nouveau lot) |
| `make predict-batch FILE=...` | prédiction par lot (`data/predictions/`) |
| `make alerts` | dernières alertes |

## API de prévision

http://localhost:8000 (documentation interactive : http://localhost:8000/docs).

| Route | Rôle |
|---|---|
| `POST /predict` | une heure → `prediction`, `model_name`, `model_version`, `prediction_id`, `latency_ms` |
| `POST /predict/batch` | jusqu'à 1 000 heures en un appel (`batch_id` + un `prediction_id` par ligne) |
| `POST /feedback` | `prediction_id` + valeur observée `actual` → erreur absolue enregistrée |
| `GET /health` | 200 `ok` / 503 `degraded` : état de la base, du modèle, dernier run de pipeline |
| `GET /model` | version servie ; `POST /model/reload` force le rechargement |

- Validation stricte : types exacts, plages du dataset, champ inconnu refusé → 422 avec la liste des
  champs en cause. Base indisponible → 503 `database_unavailable` ; modèle absent → 503
  `model_unavailable` ; erreur imprévue → 500 sans détail interne (détail dans les logs).
- Chaque prédiction (réussie ou non) est enregistrée dans PostgreSQL : entrées, sortie, version,
  latence, statut, erreur. Tables `predictions`, `feedbacks`, `pipeline_runs`, créées par la
  migration Alembic (`migrations/`, service `migrate`).
- Chaque worker uvicorn suit l'alias `champion` (toutes les 30 s) : une promotion ou un
  `make rollback` est appliqué sans redémarrage ; un registre injoignable ne coupe pas le service.

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"season":3,"yr":1,"mnth":7,"hr":17,"holiday":0,"weekday":3,"workingday":1,"weathersit":1,"temp":0.72,"atemp":0.66,"hum":0.45,"windspeed":0.19}'
```

## Supervision

| Outil | URL | Contenu |
|---|---|---|
| Grafana | http://localhost:3000 | dashboard provisionné « Bikeshare MLOps » (accès lecture sans compte ; admin/admin pour éditer) |
| Prometheus | http://localhost:9090 | métriques brutes, règles d'alerte : http://localhost:9090/alerts |
| Métriques de l'API | http://localhost:8000/metrics | format Prometheus, agrégé sur tous les workers uvicorn |

Métriques exposées : requêtes par route et code HTTP, histogramme de latence (p50/p95/p99),
temps d'inférence, prédictions par version et statut, feedbacks, erreurs par type, MAE glissante
(200 derniers feedbacks), version servie, disponibilité de la base, statut et nombre de runs de
`training_pipeline`. Règles d'alerte : `ApiDown`, `DatabaseDown`, `ModelNotLoaded`,
`HighServerErrorRate`, `HighPredictionLatencyP95`, `TrainingPipelineFailed`, `RollingMaeDegraded`.

`make traffic N=300` rejoue des heures réelles contre l'API (3 % de requêtes invalides, 80 % de
feedbacks avec la vraie valeur observée).

## Démonstrations

**`make demo-nominal`** (stack démarrée par `make up`) :
1. vérifie les services ; prépare les données si besoin ;
2. dépose le lot suivant (les 12 mois de 2011 au premier lancement, quand il n'y a pas encore de
   champion) ;
3. attend que le worker Prefect exécute `training_pipeline` et affiche la décision de promotion ;
4. attend que l'API serve le nouveau champion ;
5. envoie 300 prédictions unitaires + 5 lots, et leurs feedbacks → visibles dans Grafana.

**`make demo-incident`** (`HOLD=60` pour allonger la panne) :
- (a) lot invalide (colonne manquante, valeurs hors plage, nulls, doublons) → pipeline en échec avant
  entraînement, lot en quarantaine, alerte dans `logs/alerts.jsonl` ;
- (b) arrêt de PostgreSQL → réponses 503 `database_unavailable`, `/health` dégradé, alertes
  Prometheus `DatabaseDown` et `HighServerErrorRate`, tuile rouge dans Grafana, puis retour à la
  normale ;
- (c) arrêt de MLflow → `/model/reload` échoue proprement, le modèle en mémoire reste servi.

## Configuration

Toute la configuration est dans `config/config.yaml`. Chaque clé se surcharge par variable
d'environnement `BIKESHARE_<SECTION>__<CLE>` (voir `.env.example`), par exemple
`BIKESHARE_LOGGING__LEVEL=DEBUG`. Un autre fichier peut être utilisé via `BIKESHARE_CONFIG_FILE`.
