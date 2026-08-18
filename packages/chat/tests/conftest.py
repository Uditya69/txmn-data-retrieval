import os

import pytest

os.environ.setdefault("MONGO_URI", "mongodb://localhost:27017")
os.environ.setdefault("MONGO_DB", "test-chat-db")


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
