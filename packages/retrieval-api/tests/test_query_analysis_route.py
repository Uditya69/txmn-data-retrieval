from fastapi.testclient import TestClient

from retrieval_api.main import app


def test_query_analysis_route_returns_shape_chunks_and_es_query():
    client = TestClient(app)

    response = client.post("/v1/query-analysis", json={"query": "Section 6 of Income Tax Act"})

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "Section 6 of Income Tax Act"
    # This query's classifier confidence (~0.69) is below the confidence_threshold
    # trained by the fixed threshold sweep (0.9 - see
    # common/scripts/train_instant_classifier.py's _sweep_threshold), so
    # effective_label() correctly falls back rather than trusting a genuinely
    # uncertain HYBRID-vs-INTENT call. FALLBACK shares HYBRID's boost profile (see
    # common.instant_classifier.labels.boost_profile_key), so this doesn't change
    # the actual ES query shape.
    assert body["shape"] == "FALLBACK"
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
