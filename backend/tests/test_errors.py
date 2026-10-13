import httpx
from fastapi import FastAPI

from app.core.errors import NotFoundError


def _add_test_routes(app: FastAPI) -> None:
    @app.get("/_test/not-found")
    async def not_found() -> None:
        raise NotFoundError("Document not found.", code="DOCUMENT_NOT_FOUND", details={"id": "x"})

    @app.get("/_test/crash")
    async def crash() -> None:
        raise RuntimeError("secret internal detail")

    @app.get("/_test/validate")
    async def validate(limit: int) -> dict[str, int]:
        return {"limit": limit}


async def test_app_error_uses_envelope(app: FastAPI, client: httpx.AsyncClient) -> None:
    _add_test_routes(app)
    response = await client.get("/_test/not-found")
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "DOCUMENT_NOT_FOUND"
    assert error["message"] == "Document not found."
    assert error["details"] == {"id": "x"}
    assert error["request_id"] == response.headers["X-Request-ID"]


async def test_unknown_route_uses_envelope(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


async def test_validation_error_uses_envelope(app: FastAPI, client: httpx.AsyncClient) -> None:
    _add_test_routes(app)
    response = await client.get("/_test/validate", params={"limit": "abc"})
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "VALIDATION_ERROR"
    assert error["details"]["errors"][0]["loc"] == ["query", "limit"]


async def test_unhandled_error_hides_internals(app: FastAPI, client: httpx.AsyncClient) -> None:
    _add_test_routes(app)
    response = await client.get("/_test/crash")
    assert response.status_code == 500
    body = response.json()
    assert body["error"]["code"] == "INTERNAL_ERROR"
    assert "secret" not in response.text


async def test_incoming_request_id_is_propagated(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/health/live", headers={"X-Request-ID": "abc-123"})
    assert response.headers["X-Request-ID"] == "abc-123"


async def test_unsafe_request_id_is_replaced(client: httpx.AsyncClient) -> None:
    response = await client.get(
        "/api/health/live", headers={"X-Request-ID": "bad id\nwith newline"}
    )
    assert response.headers["X-Request-ID"] != "bad id\nwith newline"


async def test_cors_allows_configured_origin(client: httpx.AsyncClient) -> None:
    response = await client.options(
        "/api/health",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"},
    )
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"


async def test_cors_rejects_unknown_origin(client: httpx.AsyncClient) -> None:
    response = await client.options(
        "/api/health",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"},
    )
    assert "access-control-allow-origin" not in response.headers
