import httpx


class GatewayClient:
    def __init__(self, base_url: str):
        self._base_url = base_url

    async def chat(self, role: str, messages: list[dict]) -> str:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(f"{self._base_url}/v1/chat", json={"role": role, "messages": messages})
            response.raise_for_status()
            return response.json()["content"]

    async def embed(self, role: str, text: str) -> list[float]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(f"{self._base_url}/v1/embed", json={"role": role, "text": text})
            response.raise_for_status()
            return response.json()["embedding"]

    async def rerank(self, role: str, query: str, documents: list[str]) -> list[float]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self._base_url}/v1/rerank", json={"role": role, "query": query, "documents": documents}
            )
            response.raise_for_status()
            return response.json()["scores"]
