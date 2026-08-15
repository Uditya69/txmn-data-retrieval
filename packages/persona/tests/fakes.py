class FakePersonasCollection:
    """In-memory stand-in for a motor AsyncIOMotorCollection, shaped to what
    persona.repository needs: find_one and replace_one. Mirrors
    packages/auth/tests/fakes.py::FakeUsersCollection.
    """

    def __init__(self):
        self.documents: dict[str, dict] = {}

    async def find_one(self, filter: dict) -> dict | None:
        return self.documents.get(filter.get("user_id"))

    async def replace_one(self, filter: dict, replacement: dict, upsert: bool = False) -> None:
        self.documents[filter["user_id"]] = replacement
