from fastapi.testclient import TestClient

from retrieval_api.main import app


def test_query_analysis_route_returns_shape_chunks_and_es_query():
    client = TestClient(app)

    response = client.post("/v1/query-analysis", json={"query": "Section 6 of Income Tax Act"})

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "Section 6 of Income Tax Act"
    assert body["shape"] == "provision"
    assert any(c["type"] == "section" and c["text"] == "Section 6" for c in body["chunks"])
    assert "bool" in body["es_query"]


def test_query_analysis_route_does_not_execute_a_search():
    """No es_client/limit/hits anywhere in the request or response - this must never touch
    ES, only compute the breakdown (build_query_preview is pure/offline)."""
    client = TestClient(app)

    response = client.post("/v1/query-analysis", json={"query": "exemption claim"})

    body = response.json()
    assert set(body.keys()) == {"query", "shape", "expanded_query", "chunks", "es_query"}


def test_query_analysis_route_requires_query_field():
    client = TestClient(app)

    response = client.post("/v1/query-analysis", json={})

    assert response.status_code == 422
