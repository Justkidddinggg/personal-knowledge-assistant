from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chunk_overlap_smaller_than_chunk_size() -> None:
    s = get_settings()
    assert s.chunk_overlap < s.chunk_size