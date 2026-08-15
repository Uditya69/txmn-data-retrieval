import os

import pytest

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
