import httpx

_BASE_URL = "https://api.deepinfra.com/v1"


class DeepInfraAdapter:
    def __init__(self, api_key: str):
        self._headers = {"Authorization": f"Bearer {api_key}"}

    async def chat(self, model: str, messages: list[dict]) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{_BASE_URL}/openai/chat/completions",
                json={"model": model, "messages": messages},
                headers=self._headers,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

    async def embed(self, model: str, text: str) -> list[float]:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{_BASE_URL}/openai/embeddings",
                json={"model": model, "input": text},
                headers=self._headers,
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]

    async def rerank(self, model: str, query: str, documents: list[str]) -> list[float]:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{_BASE_URL}/inference/{model}",
                json={"query": query, "documents": documents},
                headers=self._headers,
            )
            response.raise_for_status()
            return response.json()["scores"]
