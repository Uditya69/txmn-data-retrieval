from fastapi.testclient import TestClient

from retrieval_api.main import app


def test_documents_endpoint_sends_cors_headers_for_browser_origin(monkeypatch):
    import retrieval_api.documents as documents_module
    from unittest.mock import AsyncMock

    async def fake_fetch_fullcontent(client, doc_id):
        return "<document><body><para>Hello.</para></body></document>"

    async def fake_fetch_document_metadata(client, doc_id):
        return {"heading": None, "subheading": None, "year": None}

    monkeypatch.setattr(documents_module, "get_settings", lambda: object())
    monkeypatch.setattr(documents_module, "get_es_client", lambda *_: AsyncMock())
    monkeypatch.setattr(documents_module, "fetch_fullcontent", fake_fetch_fullcontent)
    monkeypatch.setattr(documents_module, "fetch_document_metadata", fake_fetch_document_metadata)

    client = TestClient(app)
    response = client.get("/documents/d1", headers={"Origin": "http://localhost:8501"})

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"
