# Point d'entrée unique de toutes les commandes du projet.
# Les recettes restent simples (compatibles sh et cmd) : la logique vit dans les modules Python.

UV ?= uv
RUN := $(UV) run

.DEFAULT_GOAL := help
.PHONY: help install lint format test data next-batch invalid-batch validate train-local

COUNT ?= 1
FILES ?= data/incoming/*.csv
DATA_DIR ?= data/batches

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
	$(RUN) python -m bikeshare_mlops.models.train --data-dir $(DATA_DIR)
