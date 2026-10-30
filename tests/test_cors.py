from fastapi.testclient import TestClient


def preflight(client: TestClient, origin: str):
    """The browser sends this OPTIONS request before a cross-origin POST."""
    return client.options(
        "/ask",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )


def test_frontend_origin_is_allowed(client: TestClient) -> None:
    response = preflight(client, "http://localhost:5173")
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_other_origin_is_rejected(client: TestClient) -> None:
    response = preflight(client, "http://evil.example")
    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers