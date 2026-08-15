# Auth Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a minimal, self-contained auth service (`packages/auth`) that issues a
`user_id`-bearing JWT on signup/login, and a FastAPI dependency other services can use to resolve
`user_id` from a request without rejecting unauthenticated requests.

**Architecture:** New workspace package `packages/auth`, MongoDB-backed `users` collection.
Business logic (`service.py`) takes an injected collection object and is unit-tested against an
in-memory fake — following the existing `FakeAsyncES`/`FakeMilvusClient` pattern in
`packages/common/tests/` — never against a real Mongo instance. Thin `db.py` wires the real
`motor` (async Mongo driver) client; that wiring itself is not behavior-tested, matching how
`common/es_client.py`/`common/milvus_client.py` treat their own client construction. A FastAPI
`APIRouter` exposes `/auth/signup` and `/auth/login`; a separate dependency
(`get_current_user_id`) resolves `user_id` from an `Authorization: Bearer` header if present and
valid, returning `None` otherwise — it never raises or blocks the request.

**Tech Stack:** FastAPI, `motor` (async MongoDB driver), `bcrypt` (password hashing), `pyjwt`
(token signing), `pydantic-settings` (config) — same stack conventions as `model-gateway`/`common`.

**Spec:** `docs/superpowers/specs/2026-08-15-user-persona-system-design.md` (Auth section)

## Global Constraints

- Python 3.11, not 3.14 (repo-wide constraint — see root `CLAUDE.md` hard rule 5).
- No OAuth/social login, no email verification, no password reset, no refresh tokens, no
  server-side logout blacklist — explicitly out of scope per the spec.
- Invalid/missing auth token must **never** cause a 401/403 at the middleware/dependency layer —
  it must resolve to `user_id = None` so guest requests keep working exactly as today. Rejecting
  bad credentials happens only inside `/auth/login` itself.
- `pydantic-settings` field names must match their env var names exactly (e.g. a field
  `mongo_uri` reads `MONGO_URI`) — mismatches fail silently at runtime, not in tests, per the
  gotcha recorded in root `CLAUDE.md`.
- Package naming: Python import name is `auth` (underscore-free already), `uv` distribution name
  is `auth` in `pyproject.toml` — no dash/underscore mismatch to worry about here, unlike
  `model-gateway`/`model_gateway`.

---

## Scope note

The design spec covers two subsystems: auth (this plan) and the persona system, which depends on
auth existing. This plan covers **auth only** — it produces working, independently testable
software (signup/login/token-resolution) on its own. The persona system is a separate follow-on
plan once this one is merged.

## File Structure

```
packages/auth/
  pyproject.toml
  src/auth/
    __init__.py
    config.py       # AuthSettings (pydantic-settings): mongo_uri, jwt_secret, jwt_expiry_minutes
    security.py     # hash_password/verify_password (bcrypt), create_access_token/decode_access_token (pyjwt)
    models.py       # SignupRequest, LoginRequest, TokenResponse (pydantic)
    service.py      # signup(users, email, password), login(users, email, password) — business logic
    db.py           # get_mongo_client(settings), get_users_collection(client, settings) — real motor wiring
    router.py       # APIRouter: POST /auth/signup, POST /auth/login
    dependency.py   # get_current_user_id — FastAPI dependency for downstream services
  tests/
    conftest.py               # env var defaults, per model-gateway's conftest pattern
    fakes.py                  # FakeUsersCollection (in-memory, motor-collection-shaped)
    test_config.py
    test_security.py
    test_service.py
    test_router.py
    test_dependency.py
```

Then, wiring into the rest of the repo:

```
pyproject.toml                          # add "packages/auth" to workspace members + sources
packages/retrieval-api/pyproject.toml   # add "auth" dependency
packages/retrieval-api/src/retrieval_api/main.py  # mount auth router + middleware
docker-compose.yml                      # add `mongo` service
.env.example                            # add MONGO_URI, JWT_SECRET, JWT_EXPIRY_MINUTES
```

---

### Task 1: Package scaffolding + config

**Files:**
- Create: `packages/auth/pyproject.toml`
- Create: `packages/auth/src/auth/__init__.py`
- Create: `packages/auth/src/auth/config.py`
- Create: `packages/auth/tests/conftest.py`
- Create: `packages/auth/tests/test_config.py`

**Interfaces:**
- Produces: `AuthSettings` (pydantic-settings `BaseSettings` subclass) with fields `mongo_uri:
  str`, `mongo_db: str`, `jwt_secret: str`, `jwt_expiry_minutes: int`. `get_auth_settings() ->
  AuthSettings` (cached with `functools.lru_cache`, matching `common/config.py`'s pattern).

- [ ] **Step 1: Write `packages/auth/pyproject.toml`**

```toml
[project]
name = "auth"
version = "0.1.0"
requires-python = ">=3.11,<3.12"
dependencies = [
  "fastapi>=0.115",
  "uvicorn>=0.30",
  "motor>=3.5",
  "bcrypt>=4.2",
  "pyjwt>=2.9",
  "pydantic-settings>=2.5",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/auth"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

- [ ] **Step 2: Write `packages/auth/src/auth/__init__.py`** (empty file)

- [ ] **Step 3: Write `packages/auth/tests/conftest.py`**

```python
import os

os.environ.setdefault("MONGO_URI", "mongodb://localhost:27017")
os.environ.setdefault("MONGO_DB", "test-auth-db")
os.environ.setdefault("JWT_SECRET", "test-jwt-secret")
os.environ.setdefault("JWT_EXPIRY_MINUTES", "60")
```

- [ ] **Step 4: Write the failing test `packages/auth/tests/test_config.py`**

```python
from auth.config import get_auth_settings


def test_settings_load_from_env():
    settings = get_auth_settings()
    assert settings.mongo_uri == "mongodb://localhost:27017"
    assert settings.mongo_db == "test-auth-db"
    assert settings.jwt_secret == "test-jwt-secret"
    assert settings.jwt_expiry_minutes == 60
```

- [ ] **Step 5: Run test to verify it fails**

Run: `cd packages/auth && uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'auth.config'` (package not yet added
to the workspace, or module not yet written — run `uv sync --all-packages` from repo root first
if the import error is about the workspace member not being registered; see Task 8 for the
formal workspace-registration step, but add the member entry now so this task's tests can run):

Add `"packages/auth"` to `members` and a `auth = { workspace = true }` entry to `[tool.uv.sources]`
in the root `pyproject.toml`, then run `uv sync --all-packages` from repo root.

- [ ] **Step 6: Write minimal implementation `packages/auth/src/auth/config.py`**

```python
from functools import lru_cache

from pydantic_settings import BaseSettings


class AuthSettings(BaseSettings):
    mongo_uri: str
    mongo_db: str
    jwt_secret: str
    jwt_expiry_minutes: int


@lru_cache
def get_auth_settings() -> AuthSettings:
    return AuthSettings()
```

- [ ] **Step 7: Run test to verify it passes**

Run: `cd packages/auth && uv run pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml packages/auth
git commit -m "feat(auth): scaffold auth package with settings config"
```

---

### Task 2: Password hashing + JWT token security primitives

**Files:**
- Create: `packages/auth/src/auth/security.py`
- Create: `packages/auth/tests/test_security.py`

**Interfaces:**
- Consumes: `AuthSettings` from Task 1 (`auth.config.get_auth_settings`).
- Produces: `hash_password(password: str) -> str`, `verify_password(password: str, hashed: str)
  -> bool`, `create_access_token(user_id: str, settings: AuthSettings) -> str`,
  `decode_access_token(token: str, settings: AuthSettings) -> str | None` (returns the `user_id`
  claim, or `None` if the token is invalid/expired/malformed — never raises).

- [ ] **Step 1: Write the failing tests `packages/auth/tests/test_security.py`**

```python
import time

import jwt
import pytest

from auth.config import get_auth_settings
from auth.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


def test_hash_password_produces_different_output_and_verifies():
    hashed = hash_password("correct-horse-battery-staple")
    assert hashed != "correct-horse-battery-staple"
    assert verify_password("correct-horse-battery-staple", hashed) is True


def test_verify_password_rejects_wrong_password():
    hashed = hash_password("correct-horse-battery-staple")
    assert verify_password("wrong-password", hashed) is False


def test_create_and_decode_access_token_roundtrips_user_id():
    settings = get_auth_settings()
    token = create_access_token("user-123", settings)
    assert decode_access_token(token, settings) == "user-123"


def test_decode_access_token_rejects_garbage_token():
    settings = get_auth_settings()
    assert decode_access_token("not-a-real-token", settings) is None


def test_decode_access_token_rejects_expired_token():
    settings = get_auth_settings()
    expired_payload = {"user_id": "user-123", "exp": int(time.time()) - 10}
    expired_token = jwt.encode(expired_payload, settings.jwt_secret, algorithm="HS256")
    assert decode_access_token(expired_token, settings) is None


def test_decode_access_token_rejects_wrong_secret():
    settings = get_auth_settings()
    token = jwt.encode(
        {"user_id": "user-123", "exp": int(time.time()) + 3600},
        "a-different-secret",
        algorithm="HS256",
    )
    assert decode_access_token(token, settings) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/auth && uv run pytest tests/test_security.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'auth.security'`

- [ ] **Step 3: Write minimal implementation `packages/auth/src/auth/security.py`**

```python
import time

import bcrypt
import jwt

from auth.config import AuthSettings


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


def create_access_token(user_id: str, settings: AuthSettings) -> str:
    payload = {
        "user_id": user_id,
        "exp": int(time.time()) + settings.jwt_expiry_minutes * 60,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def decode_access_token(token: str, settings: AuthSettings) -> str | None:
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
    except jwt.PyJWTError:
        return None
    return payload.get("user_id")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/auth && uv run pytest tests/test_security.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add packages/auth/src/auth/security.py packages/auth/tests/test_security.py
git commit -m "feat(auth): add password hashing and JWT token primitives"
```

---

### Task 3: Signup/login business logic against a fake users collection

**Files:**
- Create: `packages/auth/src/auth/models.py`
- Create: `packages/auth/src/auth/service.py`
- Create: `packages/auth/tests/fakes.py`
- Create: `packages/auth/tests/test_service.py`

**Interfaces:**
- Consumes: `hash_password`, `verify_password` from Task 2 (`auth.security`).
- Produces:
  - `SignupRequest(email: str, password: str)`, `LoginRequest(email: str, password: str)`,
    `TokenResponse(access_token: str, token_type: str = "bearer")` (pydantic models in
    `auth.models`).
  - `EmailAlreadyRegistered(Exception)`, `InvalidCredentials(Exception)` (in `auth.service`).
  - `async def signup(users, email: str, password: str) -> str` — inserts a new user document
    `{"_id": <uuid4 str>, "email": email, "password_hash": ..., "created_at": <iso str>}`,
    returns the new `user_id` string. Raises `EmailAlreadyRegistered` if `email` already exists.
  - `async def login(users, email: str, password: str) -> str` — returns the matching
    `user_id` string. Raises `InvalidCredentials` if email not found or password doesn't match.
  - `users` is any object exposing `async def find_one(filter: dict) -> dict | None` and `async
    def insert_one(document: dict) -> None` — this is the shape both the test fake
    (`tests/fakes.py::FakeUsersCollection`) and the real `motor` collection (Task 4) satisfy.

- [ ] **Step 1: Write `packages/auth/src/auth/models.py`**

```python
from pydantic import BaseModel


class SignupRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
```

- [ ] **Step 2: Write `packages/auth/tests/fakes.py`**

```python
class FakeUsersCollection:
    """In-memory stand-in for a motor AsyncIOMotorCollection, shaped to what
    auth.service needs: find_one and insert_one. Mirrors the FakeAsyncES /
    FakeMilvusClient pattern used in packages/common/tests/.
    """

    def __init__(self):
        self.documents: list[dict] = []

    async def find_one(self, filter: dict) -> dict | None:
        for doc in self.documents:
            if all(doc.get(k) == v for k, v in filter.items()):
                return doc
        return None

    async def insert_one(self, document: dict) -> None:
        self.documents.append(document)
```

- [ ] **Step 3: Write the failing tests `packages/auth/tests/test_service.py`**

```python
import pytest

from auth.security import hash_password
from auth.service import (
    EmailAlreadyRegistered,
    InvalidCredentials,
    login,
    signup,
)
from tests.fakes import FakeUsersCollection


@pytest.mark.asyncio
async def test_signup_creates_user_and_returns_user_id():
    users = FakeUsersCollection()
    user_id = await signup(users, "alice@example.com", "hunter2")
    assert isinstance(user_id, str) and user_id
    stored = users.documents[0]
    assert stored["email"] == "alice@example.com"
    assert stored["_id"] == user_id
    assert stored["password_hash"] != "hunter2"


@pytest.mark.asyncio
async def test_signup_rejects_duplicate_email():
    users = FakeUsersCollection()
    await signup(users, "alice@example.com", "hunter2")
    with pytest.raises(EmailAlreadyRegistered):
        await signup(users, "alice@example.com", "different-password")


@pytest.mark.asyncio
async def test_login_returns_user_id_for_correct_credentials():
    users = FakeUsersCollection()
    user_id = await signup(users, "alice@example.com", "hunter2")
    result = await login(users, "alice@example.com", "hunter2")
    assert result == user_id


@pytest.mark.asyncio
async def test_login_rejects_unknown_email():
    users = FakeUsersCollection()
    with pytest.raises(InvalidCredentials):
        await login(users, "nobody@example.com", "hunter2")


@pytest.mark.asyncio
async def test_login_rejects_wrong_password():
    users = FakeUsersCollection()
    await signup(users, "alice@example.com", "hunter2")
    with pytest.raises(InvalidCredentials):
        await login(users, "alice@example.com", "wrong-password")
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `cd packages/auth && uv run pytest tests/test_service.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'auth.service'`

- [ ] **Step 5: Write minimal implementation `packages/auth/src/auth/service.py`**

```python
import uuid
from datetime import datetime, timezone

from auth.security import hash_password, verify_password


class EmailAlreadyRegistered(Exception):
    pass


class InvalidCredentials(Exception):
    pass


async def signup(users, email: str, password: str) -> str:
    existing = await users.find_one({"email": email})
    if existing is not None:
        raise EmailAlreadyRegistered(email)

    user_id = str(uuid.uuid4())
    await users.insert_one({
        "_id": user_id,
        "email": email,
        "password_hash": hash_password(password),
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    return user_id


async def login(users, email: str, password: str) -> str:
    user = await users.find_one({"email": email})
    if user is None or not verify_password(password, user["password_hash"]):
        raise InvalidCredentials(email)
    return user["_id"]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd packages/auth && uv run pytest tests/test_service.py -v`
Expected: PASS (5 tests)

- [ ] **Step 7: Commit**

```bash
git add packages/auth/src/auth/models.py packages/auth/src/auth/service.py packages/auth/tests/fakes.py packages/auth/tests/test_service.py
git commit -m "feat(auth): add signup/login business logic"
```

---

### Task 4: Real MongoDB wiring

**Files:**
- Create: `packages/auth/src/auth/db.py`
- Create: `packages/auth/tests/test_db.py`

**Interfaces:**
- Consumes: `AuthSettings` from Task 1.
- Produces: `get_mongo_client(settings: AuthSettings) -> AsyncIOMotorClient`,
  `get_users_collection(client: AsyncIOMotorClient, settings: AuthSettings) ->
  AsyncIOMotorCollection` (returns `client[settings.mongo_db]["users"]`).

This task's wiring is thin passthrough, not behavior — the only thing worth unit-testing is that
`get_users_collection` selects the right database/collection names, without needing a live Mongo
connection (matches how `common/es_client.py::get_es_client` construction itself isn't
behavior-tested against a live cluster).

- [ ] **Step 1: Write the failing test `packages/auth/tests/test_db.py`**

```python
from auth.config import get_auth_settings
from auth.db import get_mongo_client, get_users_collection


def test_get_users_collection_selects_configured_db_and_collection_name():
    settings = get_auth_settings()
    client = get_mongo_client(settings)
    try:
        collection = get_users_collection(client, settings)
        assert collection.name == "users"
        assert collection.database.name == settings.mongo_db
    finally:
        client.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/auth && uv run pytest tests/test_db.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'auth.db'`

- [ ] **Step 3: Write minimal implementation `packages/auth/src/auth/db.py`**

```python
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection

from auth.config import AuthSettings


def get_mongo_client(settings: AuthSettings) -> AsyncIOMotorClient:
    return AsyncIOMotorClient(settings.mongo_uri)


def get_users_collection(client: AsyncIOMotorClient, settings: AuthSettings) -> AsyncIOMotorCollection:
    return client[settings.mongo_db]["users"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/auth && uv run pytest tests/test_db.py -v`
Expected: PASS. Note: constructing `AsyncIOMotorClient` does not open a network connection
eagerly (motor/pymongo connect lazily on first operation), so this test passes without a running
MongoDB instance.

- [ ] **Step 5: Commit**

```bash
git add packages/auth/src/auth/db.py packages/auth/tests/test_db.py
git commit -m "feat(auth): wire real MongoDB client and users collection"
```

---

### Task 5: HTTP router — `/auth/signup`, `/auth/login`

**Files:**
- Create: `packages/auth/src/auth/router.py`
- Create: `packages/auth/tests/test_router.py`

**Interfaces:**
- Consumes: `SignupRequest`, `LoginRequest`, `TokenResponse` from Task 3 (`auth.models`);
  `signup`, `login`, `EmailAlreadyRegistered`, `InvalidCredentials` from Task 3 (`auth.service`);
  `create_access_token` from Task 2 (`auth.security`); `get_auth_settings` from Task 1
  (`auth.config`); `get_mongo_client`, `get_users_collection` from Task 4 (`auth.db`).
- Produces: `router: APIRouter` (importable as `auth.router.router`), with a FastAPI dependency
  `get_users_dependency` that tests override via `app.dependency_overrides` to inject
  `FakeUsersCollection` instead of a real Mongo collection.

- [ ] **Step 1: Write the failing tests `packages/auth/tests/test_router.py`**

```python
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.router import get_users_dependency, router
from tests.fakes import FakeUsersCollection


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    fake_users = FakeUsersCollection()
    app.dependency_overrides[get_users_dependency] = lambda: fake_users
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/auth && uv run pytest tests/test_router.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'auth.router'`

- [ ] **Step 3: Write minimal implementation `packages/auth/src/auth/router.py`**

```python
from fastapi import APIRouter, Depends, HTTPException

from auth.config import get_auth_settings
from auth.db import get_mongo_client, get_users_collection
from auth.models import LoginRequest, SignupRequest, TokenResponse
from auth.security import create_access_token
from auth.service import EmailAlreadyRegistered, InvalidCredentials, login, signup

router = APIRouter(prefix="/auth", tags=["auth"])


def get_users_dependency():
    settings = get_auth_settings()
    client = get_mongo_client(settings)
    return get_users_collection(client, settings)


@router.post("/signup", response_model=TokenResponse)
async def signup_route(payload: SignupRequest, users=Depends(get_users_dependency)):
    settings = get_auth_settings()
    try:
        user_id = await signup(users, payload.email, payload.password)
    except EmailAlreadyRegistered:
        raise HTTPException(status_code=409, detail="email already registered")
    token = create_access_token(user_id, settings)
    return TokenResponse(access_token=token)


@router.post("/login", response_model=TokenResponse)
async def login_route(payload: LoginRequest, users=Depends(get_users_dependency)):
    settings = get_auth_settings()
    try:
        user_id = await login(users, payload.email, payload.password)
    except InvalidCredentials:
        raise HTTPException(status_code=401, detail="invalid email or password")
    token = create_access_token(user_id, settings)
    return TokenResponse(access_token=token)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/auth && uv run pytest tests/test_router.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add packages/auth/src/auth/router.py packages/auth/tests/test_router.py
git commit -m "feat(auth): add signup/login HTTP router"
```

---

### Task 6: `get_current_user_id` dependency for downstream services

**Files:**
- Create: `packages/auth/src/auth/dependency.py`
- Create: `packages/auth/tests/test_dependency.py`

**Interfaces:**
- Consumes: `decode_access_token` from Task 2 (`auth.security`); `get_auth_settings` from Task 1
  (`auth.config`).
- Produces: `async def get_current_user_id(authorization: str | None = Header(default=None)) ->
  str | None` — a FastAPI dependency. This is the function `retrieval-api` imports in Task 7 of
  the persona plan to gate persona reads. Returns `None` for: missing header, malformed header
  (no `Bearer ` prefix), or an invalid/expired token. Never raises `HTTPException`.

- [ ] **Step 1: Write the failing tests `packages/auth/tests/test_dependency.py`**

```python
import pytest

from auth.config import get_auth_settings
from auth.dependency import get_current_user_id
from auth.security import create_access_token


@pytest.mark.asyncio
async def test_returns_user_id_for_valid_bearer_token():
    settings = get_auth_settings()
    token = create_access_token("user-123", settings)
    result = await get_current_user_id(authorization=f"Bearer {token}")
    assert result == "user-123"


@pytest.mark.asyncio
async def test_returns_none_for_missing_header():
    result = await get_current_user_id(authorization=None)
    assert result is None


@pytest.mark.asyncio
async def test_returns_none_for_malformed_header():
    result = await get_current_user_id(authorization="not-a-bearer-token")
    assert result is None


@pytest.mark.asyncio
async def test_returns_none_for_invalid_token():
    result = await get_current_user_id(authorization="Bearer garbage-token")
    assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/auth && uv run pytest tests/test_dependency.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'auth.dependency'`

- [ ] **Step 3: Write minimal implementation `packages/auth/src/auth/dependency.py`**

```python
from fastapi import Header

from auth.config import get_auth_settings
from auth.security import decode_access_token


async def get_current_user_id(authorization: str | None = Header(default=None)) -> str | None:
    if authorization is None or not authorization.startswith("Bearer "):
        return None
    token = authorization.removeprefix("Bearer ")
    settings = get_auth_settings()
    return decode_access_token(token, settings)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/auth && uv run pytest tests/test_dependency.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add packages/auth/src/auth/dependency.py packages/auth/tests/test_dependency.py
git commit -m "feat(auth): add get_current_user_id dependency for downstream services"
```

---

### Task 7: Wire auth into `retrieval-api` and infra

**Files:**
- Modify: `pyproject.toml` (root workspace members — already added in Task 1, verify present)
- Modify: `packages/retrieval-api/pyproject.toml`
- Modify: `packages/retrieval-api/src/retrieval_api/main.py`
- Modify: `docker-compose.yml`
- Modify: `.env.example`
- Create: `packages/retrieval-api/tests/test_auth_wiring.py`

**Interfaces:**
- Consumes: `router` and `get_current_user_id` from `auth` package (Tasks 5, 6).
- Produces: `retrieval-api`'s `app` now includes the auth router at `/auth/*`, and exposes
  `get_current_user_id` importable from `retrieval_api.main` for the persona plan's future use
  (persona plan is out of scope here — this task only proves the wiring works end to end).

- [ ] **Step 1: Add `"auth"` to `packages/retrieval-api/pyproject.toml` dependencies**

```toml
dependencies = [
  "fastapi>=0.115",
  "uvicorn>=0.30",
  "httpx>=0.27",
  "langchain-core>=0.3",
  "langfuse>=4.14",
  "common",
  "agents",
  "auth",
]
```

- [ ] **Step 2: Confirm root `pyproject.toml` workspace registration (from Task 1)**

```toml
[tool.uv.workspace]
members = ["packages/common", "packages/model-gateway", "packages/retrieval-api", "packages/agents", "packages/auth"]

[tool.uv.sources]
common = { workspace = true }
model-gateway = { workspace = true }
retrieval-api = { workspace = true }
agents = { workspace = true }
auth = { workspace = true }
```

Run `uv sync --all-packages` from repo root after confirming.

- [ ] **Step 3: Write the failing test `packages/retrieval-api/tests/test_auth_wiring.py`**

```python
from fastapi.testclient import TestClient

from retrieval_api.main import app


def test_auth_signup_route_is_mounted():
    client = TestClient(app)
    response = client.post("/auth/signup", json={"email": "not-a-real-flow@example.com", "password": "x"})
    # Not asserting 200 here — this test only proves the route exists and is wired,
    # not that a live Mongo is reachable in this test run.
    assert response.status_code != 404
```

- [ ] **Step 4: Run test to verify it fails**

Run: `cd packages/retrieval-api && uv run pytest tests/test_auth_wiring.py -v`
Expected: FAIL with 404 (router not yet mounted)

- [ ] **Step 5: Modify `packages/retrieval-api/src/retrieval_api/main.py`**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from auth.router import router as auth_router
from retrieval_api.ws import router
from retrieval_api.documents import router as documents_router
from retrieval_api.query_analysis import router as query_analysis_router

app = FastAPI(title="retrieval-api")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"],
)
app.include_router(router)
app.include_router(documents_router)
app.include_router(query_analysis_router)
app.include_router(auth_router)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd packages/retrieval-api && uv run pytest tests/test_auth_wiring.py -v`
Expected: PASS. (The route now exists; it may still fail internally with a Mongo connection
error if no Mongo is running locally, but the test only checks for a non-404, so it passes
either way. This is intentional — proving the wiring, not full E2E behavior, which requires the
compose stack.)

- [ ] **Step 7: Add `mongo` service to `docker-compose.yml`**

```yaml
services:
  mongo:
    image: mongo:7
    ports: ["27017:27017"]
    volumes:
      - mongo-data:/data/db

  model-gateway:
    build:
      context: .
      dockerfile: packages/model-gateway/Dockerfile
    ports: ["8001:8001"]
    env_file: .env

  retrieval-api:
    build:
      context: .
      dockerfile: packages/retrieval-api/Dockerfile
    ports: ["8010:8000"]
    env_file: .env
    environment:
      GATEWAY_URL: http://model-gateway:8001
      MONGO_URI: mongodb://mongo:27017
    depends_on: [model-gateway, mongo]

  web:
    build:
      context: .
      dockerfile: packages/web/Dockerfile
    ports: ["8501:80"]
    environment:
      WS_URL: ws://localhost:8010/ws/search
      AGENT_WS_URL: ws://localhost:8010/ws/agent
    depends_on: [retrieval-api]

volumes:
  mongo-data:
```

- [ ] **Step 8: Add auth env vars to `.env.example`**

```
# auth (packages/auth)
MONGO_URI=mongodb://localhost:27017
MONGO_DB=taxmann_auth
JWT_SECRET=
JWT_EXPIRY_MINUTES=60
```

- [ ] **Step 9: Run full test suite to confirm no regressions**

Run: `uv run pytest` from repo root
Expected: all previously-passing tests still pass, plus the new `auth` package tests and
`test_auth_wiring.py`.

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml packages/retrieval-api packages/auth docker-compose.yml .env.example
git commit -m "feat(auth): wire auth router into retrieval-api and add mongo to compose"
```

---

## Self-Review Notes

- **Spec coverage:** Auth section of the spec (endpoints, JWT-no-refresh, bcrypt hashing, Mongo
  `users` collection, non-blocking middleware behavior) is covered by Tasks 1–7. Persona-facing
  sections of the spec (persona storage, extraction pipeline) are explicitly out of scope for
  this plan — see Scope note above; a follow-on plan covers them once this merges.
- **Placeholder scan:** No TBD/TODO markers; every step has runnable code.
- **Type consistency:** `user_id` is a `str` (`uuid4` hex) end to end — `service.py` returns it,
  `security.py`'s `create_access_token`/`decode_access_token` carry it as the `user_id` JWT
  claim, `dependency.py`'s `get_current_user_id` returns the same `str | None` type the persona
  plan will consume.
