import io

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGDOC_INDEX_DIR", str(tmp_path / "index"))
    from ragdoc import api

    with TestClient(api.app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ask_before_ingest_returns_409(client):
    response = client.post("/ask", json={"question": "hello?"})
    assert response.status_code == 409


def test_ingest_then_ask(client):
    payload = b"The API gateway listens on port 8443. All traffic is TLS encrypted."
    response = client.post(
        "/ingest", files={"file": ("notes.txt", io.BytesIO(payload), "text/plain")}
    )
    assert response.status_code == 200
    assert response.json()["indexed_chunks"] >= 1

    response = client.post("/ask", json={"question": "Which port does the API gateway use?"})
    assert response.status_code == 200
    body = response.json()
    assert "8443" in body["answer"]
    assert body["sources"]


def test_unsupported_file_type_rejected(client):
    response = client.post(
        "/ingest", files={"file": ("evil.exe", io.BytesIO(b"MZ"), "application/octet-stream")}
    )
    assert response.status_code == 415
