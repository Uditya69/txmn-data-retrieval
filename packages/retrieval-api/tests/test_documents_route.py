from unittest.mock import AsyncMock
from fastapi.testclient import TestClient

from retrieval_api.main import app
import retrieval_api.documents as documents_module


def test_get_document_returns_parsed_blocks_and_header_metadata(monkeypatch):
    async def fake_fetch_fullcontent(client, doc_id):
        assert doc_id == "d1"
        return "<document><body><para>Hello world.</para></body></document>"

    async def fake_fetch_document_metadata(client, doc_id):
        assert doc_id == "d1"
        return {"heading": "[2021] 1 ITR 1 (SC)", "subheading": "A vs. B", "year": "2021"}

    es_client = AsyncMock()
    monkeypatch.setattr(documents_module, "get_settings", lambda: object())
    monkeypatch.setattr(documents_module, "get_es_client", lambda *_: es_client)
    monkeypatch.setattr(documents_module, "fetch_fullcontent", fake_fetch_fullcontent)
    monkeypatch.setattr(documents_module, "fetch_document_metadata", fake_fetch_document_metadata)

    client = TestClient(app)
    response = client.get("/documents/d1")

    assert response.status_code == 200
    assert response.json() == {
        "doc_id": "d1",
        "heading": "[2021] 1 ITR 1 (SC)",
        "subheading": "A vs. B",
        "year": "2021",
        "blocks": [{"type": "paragraph", "spans": [{"type": "text", "text": "Hello world.", "bold": False, "italic": False}]}],
    }
    es_client.close.assert_awaited_once()


def test_get_document_returns_404_when_not_found(monkeypatch):
    async def fake_fetch_fullcontent(client, doc_id):
        return None

    es_client = AsyncMock()
    monkeypatch.setattr(documents_module, "get_settings", lambda: object())
    monkeypatch.setattr(documents_module, "get_es_client", lambda *_: es_client)
    monkeypatch.setattr(documents_module, "fetch_fullcontent", fake_fetch_fullcontent)

    client = TestClient(app)
    response = client.get("/documents/missing")

    assert response.status_code == 404


def test_get_document_tolerates_missing_metadata(monkeypatch):
    async def fake_fetch_fullcontent(client, doc_id):
        return "<document><body><para>Hello world.</para></body></document>"

    async def fake_fetch_document_metadata(client, doc_id):
        return None

    es_client = AsyncMock()
    monkeypatch.setattr(documents_module, "get_settings", lambda: object())
    monkeypatch.setattr(documents_module, "get_es_client", lambda *_: es_client)
    monkeypatch.setattr(documents_module, "fetch_fullcontent", fake_fetch_fullcontent)
    monkeypatch.setattr(documents_module, "fetch_document_metadata", fake_fetch_document_metadata)

    client = TestClient(app)
    response = client.get("/documents/d1")

    assert response.status_code == 200
    body = response.json()
    assert body["heading"] is None
    assert body["subheading"] is None
    assert body["year"] is None
