from fastapi.testclient import TestClient

from retrieval_api.main import app


def test_auth_signup_route_is_mounted():
    # raise_server_exceptions=False: this environment has no live Mongo, so the
    # request may fail internally (connection error surfaced as a 500). We only
    # care that the route exists and is wired - not that the full signup flow
    # succeeds, which requires the docker-compose stack (mongo service).
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/auth/signup", json={"email": "not-a-real-flow@example.com", "password": "x"})
    # Not asserting 200 here — this test only proves the route exists and is wired,
    # not that a live Mongo is reachable in this test run.
    assert response.status_code != 404
