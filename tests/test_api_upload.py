from fastapi.testclient import TestClient

from app.config import Settings

TEXT = " ".join(f"This is sentence number {i} about topic {i % 7}." for i in range(60)).encode()


def post_file(client: TestClient, name: str, data: bytes):
    return client.post("/documents", files={"file": (name, data, "text/plain")})


def stored_files(settings: Settings) -> list[str]:
    return sorted(p.name for p in settings.upload_dir.glob("*"))


def test_upload_indexes_and_keeps_original(client: TestClient, settings: Settings) -> None:
    response = post_file(client, "note.txt", TEXT)
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "indexed"
    assert body["num_chunks"] > 1
    assert client.get("/documents").json()[0]["doc_id"] == body["doc_id"]
    assert stored_files(settings) == [f"{body['doc_id']}.txt"]


def test_duplicate_upload_returns_200(client: TestClient) -> None:
    post_file(client, "a.txt", TEXT)
    response = post_file(client, "b.txt", TEXT)
    assert response.status_code == 200
    assert response.json()["status"] == "duplicate"
    assert len(client.get("/documents").json()) == 1


def test_unsupported_file_type_is_415(client: TestClient) -> None:
    response = post_file(client, "image.png", b"fake")
    assert response.status_code == 415
    assert "not supported" in response.json()["detail"]


def test_empty_document_is_422(client: TestClient) -> None:
    assert post_file(client, "empty.txt", b"   \n ").status_code == 422


def test_directory_parts_in_filename_are_stripped(client: TestClient) -> None:
    response = post_file(client, "../../evil.txt", TEXT)
    assert response.status_code == 201
    assert response.json()["filename"] == "evil.txt"


def test_missing_file_field_is_422(client: TestClient) -> None:
    assert client.post("/documents").status_code == 422


def test_delete_also_removes_stored_original(client: TestClient, settings: Settings) -> None:
    doc_id = post_file(client, "note.txt", TEXT).json()["doc_id"]
    assert stored_files(settings) == [f"{doc_id}.txt"]
    assert client.delete(f"/documents/{doc_id}").status_code == 200
    assert stored_files(settings) == []
    assert client.get("/documents").json() == []