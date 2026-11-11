"""Gestion uniforme des erreurs : toujours un JSON ``{error, message, details}`` et un code clair.

- 422 : entrée invalide (champ manquant, type ou plage incorrects, champ inconnu) ;
- 404 / 409 : ressource absente ou conflit métier (feedback en double…) ;
- 503 : dépendance indisponible (base de données, modèle non chargé) : réessayer plus tard ;
- 500 : erreur inattendue, détaillée dans les logs, jamais dans la réponse.
"""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException

from bikeshare_mlops.api.model_service import ModelUnavailableError
from bikeshare_mlops.utils.logging import get_logger

logger = get_logger(__name__)


class ApiError(Exception):
    """Erreur métier renvoyée telle quelle au client."""

    def __init__(self, status_code: int, error: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error = error
        self.message = message


def error_body(error: str, message: str, details: list[dict[str, str]] | None = None) -> dict:
    body: dict = {"error": error, "message": message}
    if details is not None:
        body["details"] = details
    return body


def _validation_details(exc: RequestValidationError) -> list[dict[str, str]]:
    details = []
    for item in exc.errors():
        location = [str(part) for part in item.get("loc", ()) if part != "body"]
        details.append({"field": ".".join(location) or "body", "message": str(item.get("msg"))})
    return details


def install_error_handlers(app: FastAPI) -> None:
    """Branche les gestionnaires d'erreurs sur l'application."""

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        details = _validation_details(exc)
        fields = ", ".join(sorted({detail["field"] for detail in details}))
        return JSONResponse(
            status_code=422,
            content=error_body("validation_error", f"Entrée invalide : {fields}", details),
        )

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=error_body(exc.error, exc.message))

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code, content=error_body("http_error", str(exc.detail))
        )

    @app.exception_handler(ModelUnavailableError)
    async def _model(request: Request, exc: ModelUnavailableError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=error_body("model_unavailable", str(exc)),
        )

    @app.exception_handler(SQLAlchemyError)
    async def _database(request: Request, exc: SQLAlchemyError) -> JSONResponse:
        logger.error("base de données indisponible", extra={"error": type(exc).__name__})
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=error_body(
                "database_unavailable",
                "Base de données indisponible : la prédiction n'a pas pu être enregistrée.",
            ),
        )

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("erreur inattendue", extra={"path": request.url.path})
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_body("internal_error", "Erreur interne inattendue."),
        )
