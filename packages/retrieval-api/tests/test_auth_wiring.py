from fastapi.testclient import TestClient

from auth.router import get_users_dependency
from retrieval_api.main import app


class _FakeUsersCollection:
    """Minimal in-test fake standing in for a motor AsyncIOMotorCollection -
    just enough shape (async find_one/insert_one) for the signup route to run
    without touching a real Mongo instance. Mirrors the FakeUsersCollection
    pattern in packages/auth/tests/conftest.py, duplicated locally since this
    test only needs the route to be reachable, not the full auth test fixture set.
    """

    def __init__(self):
        self.documents: list[dict] = []

    async def find_one(self, filter: dict):
        for doc in self.documents:
            if all(doc.get(k) == v for k, v in filter.items()):
                return doc
        return None

    async def insert_one(self, document: dict) -> None:
        self.documents.append(document)


def test_auth_signup_route_is_mounted():
    # Overrides the users dependency with an in-memory fake so this test proves
    # the route is mounted and wired, without making a real network call to Mongo
    # (a real write attempt if reachable, a slow timeout if not).
    app.dependency_overrides[get_users_dependency] = lambda: _FakeUsersCollection()
    try:
        client = TestClient(app)
        response = client.post(
            "/auth/signup", json={"email": "not-a-real-flow@example.com", "password": "x"}
        )
        assert response.status_code == 200
    finally:
        app.dependency_overrides.pop(get_users_dependency, None)
