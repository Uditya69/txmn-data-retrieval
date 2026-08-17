import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.router import get_users_dependency, router


@pytest.fixture
def client(fake_users_collection):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_users_dependency] = lambda: fake_users_collection
    return TestClient(app)


def test_signup_returns_access_token(client):
    response = client.post("/auth/signup", json={"email": "alice@example.com", "password": "hunter2"})
    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert isinstance(body["access_token"], str) and body["access_token"]


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
