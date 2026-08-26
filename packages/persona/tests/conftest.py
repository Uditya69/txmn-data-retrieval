import os

import pytest

from persona.config import get_persona_settings

os.environ.setdefault("MONGO_URI", "mongodb://localhost:27017")
os.environ.setdefault("MONGO_DB", "test-auth-db")


class FakePersonasCollection:
	"""In-memory stand-in for a motor AsyncIOMotorCollection, shaped to what
	persona.repository needs: find_one and replace_one.
	"""

	def __init__(self):
		self.documents: dict[str, dict] = {}

	async def find_one(self, filter: dict) -> dict | None:
		return self.documents.get(filter.get("user_id"))

	async def replace_one(self, filter: dict, replacement: dict, upsert: bool = False) -> None:
		self.documents[filter["user_id"]] = replacement


@pytest.fixture
def fake_personas_collection():
	return FakePersonasCollection()


class _AsyncCursor:
	def __init__(self, items):
		self._items = list(items)

	def __aiter__(self):
		return self._iterate()

	async def _iterate(self):
		for item in self._items:
			yield item


class FakeEventsCollection:
	"""In-memory stand-in for the append-only `persona_events` collection."""

	def __init__(self):
		self.documents: list[dict] = []

	async def insert_one(self, doc: dict) -> None:
		self.documents.append(dict(doc))

	def find(self, filter: dict):
		def matches(doc):
			return all(doc.get(key) == value for key, value in filter.items())
		return _AsyncCursor(doc for doc in self.documents if matches(doc))


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
		return _AsyncCursor(dict(doc) for doc in self.documents.values() if matches(doc))

	async def replace_one(self, filter: dict, replacement: dict, upsert: bool = False) -> None:
		self.documents[(filter["user_id"], filter["topic_id"])] = dict(replacement)


@pytest.fixture
def fake_topics_collection():
	return FakeTopicsCollection()


@pytest.fixture
def persona_settings():
	return get_persona_settings()
