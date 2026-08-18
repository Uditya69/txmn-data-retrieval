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


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class _FakeCursor:
    def __init__(self, items):
        self._items = items

    def __aiter__(self):
        return self._agen()

    async def _agen(self):
        for item in self._items:
            yield item


class FakeSemanticCacheCollection:
    """In-memory stand-in for the Atlas $vectorSearch-backed collection.

    Mirrors packages/semantic_cache/tests/conftest.py's fixture of the same
    name - duplicated here rather than imported, matching this file's existing
    convention for fake_conversations_collection/fake_personas_collection
    (each package's test fakes live directly in its own tests/conftest.py,
    not imported cross-package).
    """

    def __init__(self):
        self.documents: list[dict] = []

    async def insert_one(self, document: dict) -> None:
        self.documents.append(document)

    def aggregate(self, pipeline: list[dict]):
        stage = pipeline[0]["$vectorSearch"]
        query_vector = stage["queryVector"]
        mode_filter = stage.get("filter", {}).get("mode")
        limit = stage.get("limit", 1)

        candidates = [
            doc for doc in self.documents
            if mode_filter is None or doc["mode"] == mode_filter
        ]
        scored = [
            {**doc, "score": _cosine_similarity(query_vector, doc["query_embedding"])}
            for doc in candidates
        ]
        scored.sort(key=lambda d: d["score"], reverse=True)
        return _FakeCursor(scored[:limit])


@pytest.fixture
def fake_semantic_cache_collection():
    return FakeSemanticCacheCollection()
