class FakeUsersCollection:
    """In-memory stand-in for a motor AsyncIOMotorCollection, shaped to what
    auth.service needs: find_one and insert_one. Mirrors the FakeAsyncES /
    FakeMilvusClient pattern used in packages/common/tests/.
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
