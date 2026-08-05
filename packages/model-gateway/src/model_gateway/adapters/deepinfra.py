import httpx

_BASE_URL = "https://api.deepinfra.com/v1"


def _openai_usage_details(usage: dict) -> dict[str, int]:
    """Map an OpenAI-shaped usage block to Langfuse's usage_details keys."""
    details = {}
    if "prompt_tokens" in usage:
        details["input"] = usage["prompt_tokens"]
    if "completion_tokens" in usage:
        details["output"] = usage["completion_tokens"]
    return details


class DeepInfraAdapter:
    def __init__(self, api_key: str):
        self._headers = {"Authorization": f"Bearer {api_key}"}

    async def chat(self, model: str, messages: list[dict]) -> tuple[str, dict[str, int], str | None]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{_BASE_URL}/openai/chat/completions",
                json={"model": model, "messages": messages},
                headers=self._headers,
            )
            response.raise_for_status()
            data = response.json()
            usage = data.get("usage") or {}
            message = data["choices"][0]["message"]
            # Reasoning models (DeepSeek-R1, QwQ, ...) expose their chain of
            # thought as a separate field alongside the final content.
            return message["content"], _openai_usage_details(usage), message.get("reasoning_content")

    async def embed(self, model: str, text: str) -> tuple[list[float], dict[str, int]]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{_BASE_URL}/openai/embeddings",
                json={"model": model, "input": text},
                headers=self._headers,
            )
            response.raise_for_status()
            data = response.json()
            usage = data.get("usage") or {}
            return data["data"][0]["embedding"], _openai_usage_details(usage)

    async def rerank(self, model: str, query: str, documents: list[str]) -> list[float]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{_BASE_URL}/inference/{model}",
                json={"queries": [query], "documents": documents},
                headers=self._headers,
            )
            response.raise_for_status()
            return response.json()["scores"]
