import os

os.environ.setdefault("MONGO_URI", "mongodb://localhost:27017")
os.environ.setdefault("MONGO_DB", "test-auth-db")
os.environ.setdefault("JWT_SECRET", "test-jwt-secret")
os.environ.setdefault("JWT_EXPIRY_MINUTES", "60")

import pytest


class FakeUsersCollection:
    """In-memory stand-in for a motor AsyncIOMotorCollection, shaped to what
    auth.service needs: find_one and insert_one. Mirrors the FakeAsyncES /
    FakeMilvusClient pattern used in packages/common/tests/.

    Defined in conftest.py (rather than a separate tests/fakes.py module) so it's
    reachable without a cross-file import - this tests/ directory has no unique
    package identity when the full multi-package suite is collected from the repo
    root (import-mode=importlib resolves collisions between packages' own tests/
    directories, but a plain `from fakes import ...`/`from tests.fakes import ...`
    still depends on sys.path or package-name assumptions that don't hold there).
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


@pytest.fixture
def fake_users_collection():
    return FakeUsersCollection()
