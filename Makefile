# Point d'entrée unique de toutes les commandes du projet.
# Les recettes restent simples (compatibles sh et cmd) : la logique vit dans les modules Python.

UV ?= uv
RUN := $(UV) run

.DEFAULT_GOAL := help
.PHONY: help install lint format test

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
