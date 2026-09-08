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
        "blocks": [{"type": "paragraph", "spans": [
            {"type": "text", "text": "Hello world.", "bold": False, "italic": False, "highlight": False},
        ]}],
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


def test_get_document_with_query_uses_highlighted_fetch(monkeypatch):
    """Prod parity: the reader passes the query it was opened from so the
    server can highlight real matches (fetch_highlighted_fullcontent), not
    the frontend re-guessing with a client-side regex against plain text."""
    async def fake_fetch_highlighted_fullcontent(client, doc_id, query):
        assert doc_id == "d1"
        assert query == "return filed"
        return "<document><body><para>Assessee <mark>filed</mark> a return.</para></body></document>"

    async def fake_fetch_fullcontent(client, doc_id):
        raise AssertionError("plain fetch_fullcontent must not be used when a query is given")

    async def fake_fetch_document_metadata(client, doc_id):
        return {"heading": "h", "subheading": "s", "year": "2021"}

    es_client = AsyncMock()
    monkeypatch.setattr(documents_module, "get_settings", lambda: object())
    monkeypatch.setattr(documents_module, "get_es_client", lambda *_: es_client)
    monkeypatch.setattr(documents_module, "fetch_fullcontent", fake_fetch_fullcontent)
    monkeypatch.setattr(documents_module, "fetch_highlighted_fullcontent", fake_fetch_highlighted_fullcontent)
    monkeypatch.setattr(documents_module, "fetch_document_metadata", fake_fetch_document_metadata)

    client = TestClient(app)
    response = client.get("/documents/d1", params={"query": "return filed"})

    assert response.status_code == 200
    spans = response.json()["blocks"][0]["spans"]
    assert spans == [
        {"type": "text", "text": "Assessee ", "bold": False, "italic": False, "highlight": False},
        {"type": "text", "text": "filed", "bold": False, "italic": False, "highlight": True},
        {"type": "text", "text": " a return.", "bold": False, "italic": False, "highlight": False},
    ]


def test_get_document_without_query_uses_plain_fetch(monkeypatch):
    """No query context (e.g. a bookmarked/direct link) - unchanged behavior,
    plain fullcontent, no highlighting."""
    async def fake_fetch_highlighted_fullcontent(client, doc_id, query):
        raise AssertionError("fetch_highlighted_fullcontent must not be used without a query")

    async def fake_fetch_fullcontent(client, doc_id):
        return "<document><body><para>Hello world.</para></body></document>"

    async def fake_fetch_document_metadata(client, doc_id):
        return {"heading": "h", "subheading": "s", "year": "2021"}

    es_client = AsyncMock()
    monkeypatch.setattr(documents_module, "get_settings", lambda: object())
    monkeypatch.setattr(documents_module, "get_es_client", lambda *_: es_client)
    monkeypatch.setattr(documents_module, "fetch_fullcontent", fake_fetch_fullcontent)
    monkeypatch.setattr(documents_module, "fetch_highlighted_fullcontent", fake_fetch_highlighted_fullcontent)
    monkeypatch.setattr(documents_module, "fetch_document_metadata", fake_fetch_document_metadata)

    client = TestClient(app)
    response = client.get("/documents/d1")

    assert response.status_code == 200


def test_get_document_falls_back_to_plain_fullcontent_on_malformed_highlighted_xml(monkeypatch):
    """A phrase-highlight match that straddles an XML element boundary could
    (rarely) produce malformed XML - degrade the same way genuinely malformed
    source XML already does (strip_tags_fallback), never a 500."""
    async def fake_fetch_highlighted_fullcontent(client, doc_id, query):
        return "<document><body><para>Broken <mark>tag</para></body></document>"

    async def fake_fetch_document_metadata(client, doc_id):
        return {"heading": "h", "subheading": "s", "year": "2021"}

    es_client = AsyncMock()
    monkeypatch.setattr(documents_module, "get_settings", lambda: object())
    monkeypatch.setattr(documents_module, "get_es_client", lambda *_: es_client)
    monkeypatch.setattr(documents_module, "fetch_highlighted_fullcontent", fake_fetch_highlighted_fullcontent)
    monkeypatch.setattr(documents_module, "fetch_document_metadata", fake_fetch_document_metadata)

    client = TestClient(app)
    response = client.get("/documents/d1", params={"query": "tag"})

    assert response.status_code == 200
    assert "Broken" in response.json()["blocks"][0]["spans"][0]["text"]


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
