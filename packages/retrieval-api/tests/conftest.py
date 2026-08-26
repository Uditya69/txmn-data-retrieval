import os

import pytest
from pymongo.errors import DuplicateKeyError

os.environ.setdefault(
    "MONGO_URI",
    "mongodb://localhost:27017/?serverSelectionTimeoutMS=2000&connectTimeoutMS=2000",
)
os.environ.setdefault("MONGO_DB", "test-auth-db")
os.environ.setdefault("JWT_SECRET", "test-jwt-secret-that-is-at-least-32-characters-long")
os.environ.setdefault("JWT_EXPIRY_MINUTES", "60")
# Dummy env vars for common.Settings required fields
os.environ.setdefault("MILVUS_URI", "http://localhost:19530")
os.environ.setdefault("MILVUS_TOKEN", "test-milvus-token")
os.environ.setdefault("ES_URI", "http://localhost:9200")
os.environ.setdefault("GATEWAY_URL", "http://localhost:8000")


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


class FakeEventsCollection:
    """In-memory stand-in for the append-only `persona_events` collection -
    mirrors packages/persona/tests/conftest.py's fixture of the same shape,
    duplicated here per this file's existing per-package fake convention."""

    def __init__(self):
        self.documents: list[dict] = []

    async def insert_one(self, doc: dict) -> None:
        self.documents.append(dict(doc))

    def find(self, filter: dict):
        def matches(doc):
            return all(doc.get(key) == value for key, value in filter.items())
        return _FakeCursor([doc for doc in self.documents if matches(doc)])


@pytest.fixture
def fake_events_collection():
    return FakeEventsCollection()


class FakeTopicsCollection:
    """In-memory stand-in for the derived/cache `persona_topics` collection."""

    def __init__(self):
        self.documents: dict[tuple, dict] = {}

    def find(self, filter: dict):
        def matches(doc):
            return all(doc.get(key) == value for key, value in filter.items())
        return _FakeCursor([dict(doc) for doc in self.documents.values() if matches(doc)])

    async def replace_one(self, filter: dict, replacement: dict, upsert: bool = False) -> None:
        self.documents[(filter["user_id"], filter["topic_id"])] = dict(replacement)


@pytest.fixture
def fake_topics_collection():
    return FakeTopicsCollection()


@pytest.fixture
def persona_settings():
    from persona.config import get_persona_settings
    return get_persona_settings()


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


class FakeConversationsCollection:
    """In-memory stand-in for a motor AsyncIOMotorCollection, shaped to what
    chat.repository needs: find_one, replace_one, find (list), delete_one.
    """

    def __init__(self):
        self.documents: dict[str, dict] = {}

    async def find_one(self, filter: dict) -> dict | None:
        doc = self.documents.get(filter.get("_id"))
        if doc is None:
            return None
        if "user_id" in filter and doc.get("user_id") != filter["user_id"]:
            return None
        return doc

    async def replace_one(self, filter: dict, replacement: dict, upsert: bool = False) -> None:
        # `_id` is a real unique index in Mongo: if a document with this _id
        # already exists but doesn't satisfy the rest of the filter (e.g. it
        # belongs to a different user_id), an upsert can't "match no
        # document and insert a new one" - Mongo would try to insert a new
        # doc with the same _id and hit the unique index, raising
        # DuplicateKeyError instead of silently overwriting someone else's
        # document. Mirror that here rather than the old behavior of keying
        # blindly off filter["_id"] regardless of the rest of the filter.
        existing = self.documents.get(filter["_id"])
        if existing is not None:
            if all(existing.get(k) == v for k, v in filter.items()):
                self.documents[filter["_id"]] = replacement
                return
            if upsert:
                raise DuplicateKeyError(f"E11000 duplicate key error, _id: {filter['_id']!r}")
            return
        self.documents[filter["_id"]] = replacement

    def find(self, filter: dict):
        matches = [d for d in self.documents.values() if d.get("user_id") == filter.get("user_id")]

        class _Cursor:
            def __init__(self, docs):
                self._docs = docs

            def sort(self, field, direction):
                reverse = direction < 0
                self._docs = sorted(self._docs, key=lambda d: d[field], reverse=reverse)
                return self

            def __aiter__(self):
                self._iter = iter(self._docs)
                return self

            async def __anext__(self):
                try:
                    return next(self._iter)
                except StopIteration:
                    raise StopAsyncIteration

        return _Cursor(matches)

    async def delete_one(self, filter: dict) -> "_DeleteResult":
        doc = self.documents.get(filter.get("_id"))
        deleted = 0
        if doc is not None and doc.get("user_id") == filter.get("user_id"):
            del self.documents[filter["_id"]]
            deleted = 1
        return _DeleteResult(deleted)


class _DeleteResult:
    def __init__(self, deleted_count: int):
        self.deleted_count = deleted_count


@pytest.fixture
def fake_conversations_collection():
    return FakeConversationsCollection()
