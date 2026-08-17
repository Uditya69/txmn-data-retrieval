import os

import pytest

os.environ.setdefault(
    "MONGO_URI",
    "mongodb://localhost:27017/?serverSelectionTimeoutMS=2000&connectTimeoutMS=2000",
)
os.environ.setdefault("MONGO_DB", "test-auth-db")
os.environ.setdefault("JWT_SECRET", "test-jwt-secret-that-is-at-least-32-characters-long")
os.environ.setdefault("JWT_EXPIRY_MINUTES", "60")


class FakePersonasCollection:
    """In-memory stand-in for a Mongo personas collection, used by
    test_persona_signal.py. Defined here (not in a separately-imported
    tests/tests_persona_fakes.py module) because this repo's aggregated
    repo-root `uv run pytest` run always applies --import-mode=importlib
    (packages/retrieval-api's own pyproject.toml has no
    [tool.pytest.ini_options] section, so pytest walks up and picks up the
    repo root's ini config even for a package-scoped run from
    packages/retrieval-api) - under importlib mode, tests/ is never added to
    sys.path, so a plain `import tests_persona_fakes` fails identically at
    both the package-level and aggregated-suite invocations. A pytest
    fixture sidesteps this entirely."""

    def __init__(self):
        self.documents: dict[str, dict] = {}

    async def find_one(self, filter: dict) -> dict | None:
        return self.documents.get(filter.get("user_id"))

    async def replace_one(self, filter: dict, replacement: dict, upsert: bool = False) -> None:
        self.documents[filter["user_id"]] = replacement


@pytest.fixture
def fake_personas_collection():
    return FakePersonasCollection()
