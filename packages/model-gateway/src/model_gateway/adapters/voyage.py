import httpx

_BASE_URL = "https://api.voyageai.com/v1"


class VoyageAdapter:
    """Embed-only adapter — Voyage is used exclusively for query_embed, to
    stay in the same vector space as data-extraction-pipeline's Voyage-embedded
    `dense_vector` corpus. See Task 6 note in the plan for why this must not
    be swapped for another embedding provider."""

    def __init__(self, api_key: str):
        self._headers = {"Authorization": f"Bearer {api_key}"}

    async def chat(self, model: str, messages: list[dict]) -> str:
        raise NotImplementedError("VoyageAdapter does not support chat")

    async def embed(self, model: str, text: str) -> list[float]:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{_BASE_URL}/embeddings",
                json={"input": [text], "model": model, "input_type": "query"},
                headers=self._headers,
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]

    async def rerank(self, model: str, query: str, documents: list[str]) -> list[float]:
        raise NotImplementedError("VoyageAdapter does not support rerank")
