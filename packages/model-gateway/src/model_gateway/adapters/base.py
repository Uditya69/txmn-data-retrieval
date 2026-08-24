from typing import Protocol


class ModelAdapter(Protocol):
    async def chat(
        self, model: str, messages: list[dict], response_format: dict | None = None,
        temperature: float | None = None,
    ) -> tuple[str | None, dict[str, int], str | None]: ...
    async def embed(self, model: str, text: str) -> list[float]: ...
    async def rerank(self, model: str, query: str, documents: list[str]) -> list[float]: ...
