import os

os.environ.setdefault("MONGO_URI", "mongodb://localhost:27017")
os.environ.setdefault("MONGO_DB", "test-semantic-cache-db")

import pytest


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

    Simulates Atlas's $vectorSearch aggregation stage via brute-force cosine
    similarity, since there is no local Atlas cluster to test against and
    this repo's convention (see packages/persona/tests/conftest.py) is a
    hand-rolled in-memory fake rather than mongomock or a real test Mongo.
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
