import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.router import get_refresh_tokens_dependency, get_users_dependency, router


@pytest.fixture
def client(fake_users_collection, fake_refresh_tokens_collection):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_users_dependency] = lambda: fake_users_collection
    app.dependency_overrides[get_refresh_tokens_dependency] = lambda: fake_refresh_tokens_collection
    return TestClient(app)


def test_signup_returns_access_token(client):
    response = client.post("/auth/signup", json={"email": "alice@example.com", "password": "hunter2"})
    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert isinstance(body["access_token"], str) and body["access_token"]
    assert isinstance(body["refresh_token"], str) and body["refresh_token"]


def test_signup_rejects_duplicate_email(client):
    client.post("/auth/signup", json={"email": "alice@example.com", "password": "hunter2"})
    response = client.post("/auth/signup", json={"email": "alice@example.com", "password": "other"})
    assert response.status_code == 409


def test_login_returns_access_token_for_correct_credentials(client):
    client.post("/auth/signup", json={"email": "alice@example.com", "password": "hunter2"})
    response = client.post("/auth/login", json={"email": "alice@example.com", "password": "hunter2"})
    assert response.status_code == 200
    assert response.json()["token_type"] == "bearer"


def test_login_rejects_wrong_password(client):
    client.post("/auth/signup", json={"email": "alice@example.com", "password": "hunter2"})
    response = client.post("/auth/login", json={"email": "alice@example.com", "password": "wrong"})
    assert response.status_code == 401


def test_signup_rejects_password_over_72_bytes(client):
    long_password = "a" * 73
    response = client.post("/auth/signup", json={"email": "bob@example.com", "password": long_password})
    assert response.status_code == 422


def test_login_returns_refresh_token(client):
    client.post("/auth/signup", json={"email": "alice@example.com", "password": "hunter2"})
    response = client.post("/auth/login", json={"email": "alice@example.com", "password": "hunter2"})
    assert isinstance(response.json()["refresh_token"], str) and response.json()["refresh_token"]


def test_refresh_returns_a_new_token_pair(client):
    # Not asserting access_token != signup_body["access_token"]: a JWT minted
    # for the same user_id in the same wall-clock second (exp granularity is
    # seconds) is legitimately byte-identical - that's fine functionally, just
    # not a property this test can rely on. The refresh_token has no such
    # coincidence risk (32 bytes of CSPRNG output), so that's the real check.
    signup_body = client.post("/auth/signup", json={"email": "alice@example.com", "password": "hunter2"}).json()

    response = client.post("/auth/refresh", json={"refresh_token": signup_body["refresh_token"]})

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["access_token"], str) and body["access_token"]
    assert body["refresh_token"] != signup_body["refresh_token"]


def test_refresh_rejects_an_already_used_token(client):
    signup_body = client.post("/auth/signup", json={"email": "alice@example.com", "password": "hunter2"}).json()
    client.post("/auth/refresh", json={"refresh_token": signup_body["refresh_token"]})

    response = client.post("/auth/refresh", json={"refresh_token": signup_body["refresh_token"]})

    assert response.status_code == 401


def test_refresh_rejects_unknown_token(client):
    response = client.post("/auth/refresh", json={"refresh_token": "never-issued"})
    assert response.status_code == 401


def test_logout_revokes_the_refresh_token(client):
    signup_body = client.post("/auth/signup", json={"email": "alice@example.com", "password": "hunter2"}).json()

    logout_response = client.post("/auth/logout", json={"refresh_token": signup_body["refresh_token"]})
    assert logout_response.status_code == 204

    refresh_response = client.post("/auth/refresh", json={"refresh_token": signup_body["refresh_token"]})
    assert refresh_response.status_code == 401


def test_logout_does_not_error_on_an_unknown_token(client):
    response = client.post("/auth/logout", json={"refresh_token": "never-issued"})
    assert response.status_code == 204
