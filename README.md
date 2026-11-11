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

## Configuration

Toute la configuration est dans `config/config.yaml`. Chaque clé se surcharge par variable
d'environnement `BIKESHARE_<SECTION>__<CLE>` (voir `.env.example`), par exemple
`BIKESHARE_LOGGING__LEVEL=DEBUG`. Un autre fichier peut être utilisé via `BIKESHARE_CONFIG_FILE`.
