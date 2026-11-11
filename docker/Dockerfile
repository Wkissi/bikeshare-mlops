# Image applicative unique (MLflow, Prefect, pipeline, API) : mêmes versions figées partout (uv.lock).
FROM python:3.11-slim

# Commit Git du code embarqué (fourni par `make up`), tracé dans chaque run MLflow.
ARG CODE_VERSION=unknown

ENV CODE_VERSION=${CODE_VERSION} \
    GIT_PYTHON_REFRESH=quiet \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH

COPY --from=ghcr.io/astral-sh/uv:0.11.6 /uv /usr/local/bin/uv

WORKDIR /app

# Dépendances d'abord : cette couche reste en cache tant que uv.lock ne change pas.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY config ./config
COPY scripts ./scripts
COPY alembic.ini ./
COPY migrations ./migrations
RUN uv sync --frozen --no-dev

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data /app/reports /app/logs /mlartifacts \
    && chown -R app:app /app /mlartifacts
USER app
