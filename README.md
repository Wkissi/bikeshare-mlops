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

## Configuration

Toute la configuration est dans `config/config.yaml`. Chaque clé se surcharge par variable
d'environnement `BIKESHARE_<SECTION>__<CLE>` (voir `.env.example`), par exemple
`BIKESHARE_LOGGING__LEVEL=DEBUG`. Un autre fichier peut être utilisé via `BIKESHARE_CONFIG_FILE`.
