import httpx


class LocalRerankAdapter:
    """Self-hosted cross-encoder reranker (Qwen3-Reranker served on its own port, separate
    from LocalAdapter's chat-completions host) - rerank-only, mirrors VoyageAdapter's
    embed-only shape. Its /rerank response is `{"results": [{"index", "relevance_score"}]}`,
    sorted by score rather than input order - unlike DeepInfraAdapter's /inference/{model}
    (`{"scores": [...]}`, already aligned to input document order). Reordered by `index`
    here so every adapter's rerank() honors the same contract: a list of scores positionally
    aligned to the `documents` argument, which routes.py's /v1/rerank then zips straight
    back against req.documents."""

    def __init__(self, base_url: str, api_key: str):
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"}

    async def chat(
        self, model: str, messages: list[dict], response_format: dict | None = None,
        temperature: float | None = None,
    ) -> tuple[str | None, dict[str, int], str | None]:
        raise NotImplementedError("LocalRerankAdapter does not support chat")

    async def embed(self, model: str, text: str) -> tuple[list[float], dict[str, int]]:
        raise NotImplementedError("LocalRerankAdapter does not support embed")

    async def rerank(self, model: str, query: str, documents: list[str]) -> list[float]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self._base_url}/rerank",
                json={"model": model, "query": query, "documents": documents},
                headers=self._headers,
            )
            response.raise_for_status()
            results = response.json()["results"]
            by_index = sorted(results, key=lambda row: row["index"])
            return [row["relevance_score"] for row in by_index]
