# Point d'entrée unique de toutes les commandes du projet.
# Les recettes restent simples (compatibles sh et cmd) : la logique vit dans les modules Python.

UV ?= uv
RUN := $(UV) run

.DEFAULT_GOAL := help
.PHONY: help install lint format test data next-batch invalid-batch validate train-local \
	up down ps logs train promote rollback models pipeline predict-batch alerts

COUNT ?= 1
FILES ?= data/incoming/*.csv
COMPOSE := docker compose
export CODE_VERSION := $(shell git rev-parse --short HEAD)
train-local: DATA_DIR ?= data/batches
train promote: DATA_DIR ?= data/processed
pipeline predict-batch: export PREFECT_API_URL := http://localhost:4200/api

help: ## Liste les commandes disponibles
	@$(RUN) python scripts/make_help.py Makefile

install: ## Installe Python 3.11 et les dépendances figées (uv.lock)
	$(UV) python install 3.11
	$(UV) sync --frozen --all-groups

lint: ## Vérifie le style et les erreurs courantes (ruff)
	$(RUN) ruff check src tests scripts
	$(RUN) ruff format --check src tests scripts

format: ## Reformate le code et corrige ce qui peut l'être automatiquement
	$(RUN) ruff check --fix src tests scripts
	$(RUN) ruff format src tests scripts

test: ## Lance tous les tests avec la couverture
	$(RUN) pytest

data: ## Télécharge la source UCI (checksum vérifié) et la découpe en lots mensuels
	$(RUN) python -m bikeshare_mlops.data.ingest prepare

next-batch: ## Fait arriver le(s) prochain(s) lot(s) dans data/incoming (COUNT=n)
	$(RUN) python -m bikeshare_mlops.data.ingest release --count $(COUNT)

invalid-batch: ## Dépose un lot volontairement non conforme dans data/incoming
	$(RUN) python scripts/make_invalid_batch.py

validate: ## Contrôle qualité de fichiers de lots (FILES="chemin1 chemin2")
	$(RUN) python -m bikeshare_mlops.data.validate $(wildcard $(FILES))

train-local: ## Entraîne et évalue hors MLflow sur DATA_DIR (défaut : tous les lots)
	$(RUN) python -m bikeshare_mlops.models.train --no-mlflow --data-dir $(DATA_DIR)

up: ## Construit et démarre les services Docker (attend qu'ils soient sains)
	$(COMPOSE) up -d --build --wait

down: ## Arrête les services (les volumes sont conservés)
	$(COMPOSE) down

ps: ## État des services Docker
	$(COMPOSE) ps

logs: ## Journaux des services (SERVICE=nom pour un seul)
	$(COMPOSE) logs --tail 100 $(SERVICE)

train: ## Entraîne sur DATA_DIR, trace dans MLflow et enregistre le challenger
	$(RUN) python -m bikeshare_mlops.models.train --data-dir $(DATA_DIR)

promote: ## Compare le challenger au champion et applique les règles de promotion
	$(RUN) python -m bikeshare_mlops.models.promote decide --data-dir $(DATA_DIR)

rollback: ## Remet l'alias champion sur une version précédente (VERSION=n)
	$(RUN) python -m bikeshare_mlops.models.promote rollback --version $(VERSION)

models: ## État du registre : versions, alias, statut de promotion
	$(RUN) python -m bikeshare_mlops.models.registry

pipeline: ## Exécute une fois le flow training_pipeline depuis l'hôte (RETRAIN=1 : sans nouveau lot)
	$(RUN) python -m bikeshare_mlops.pipelines.training $(if $(RETRAIN),--retrain,)

predict-batch: ## Prédit un CSV avec le champion (FILE=chemin)
	$(RUN) python -m bikeshare_mlops.pipelines.batch_prediction $(FILE)

alerts: ## Affiche les dernières alertes
	$(RUN) python -m bikeshare_mlops.utils.alerts
