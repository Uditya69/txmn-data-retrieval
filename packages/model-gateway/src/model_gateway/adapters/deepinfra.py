import httpx

_BASE_URL = "https://api.deepinfra.com/v1"

# DeepInfra defaults an unset max_tokens to a value derived from the largest
# model it serves (observed: 65536), which exceeds smaller models' own
# context limits (e.g. Qwen3-30B-A3B's 40960) and 400s.
#
# 32768 is sized to stay under the smallest context limit among currently
# configured models (40960, shared by Qwen3-235B-A22B and Qwen3-30B-A3B; the
# Llama-3.1 8B/70B roles allow 131072). It is NOT a "safe for every role"
# constant: reasoning-heavy models (e.g. the deployed agent_chat role,
# Qwen3-235B-A22B, which always emits reasoning_content) count reasoning
# tokens against this budget, and this branch's own eval runs all used
# --skip-agentic, so agent_chat was never exercised against this cap. A
# reasoning turn that burns through the budget yields empty content, which
# agents/loop.py silently treats as "the model is done" - no error raised.
# Treat this as a conservative headroom increase over the old 4096, not a
# proven-safe value for arbitrarily long reasoning traces.
_CHAT_MAX_TOKENS = 32768


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

    async def chat(
        self, model: str, messages: list[dict], tools: list[dict] | None = None,
        tool_choice: str | None = None, response_format: dict | None = None,
    ) -> tuple[str | None, dict[str, int], str | None, list[dict] | None]:
        payload = {"model": model, "messages": messages, "max_tokens": _CHAT_MAX_TOKENS}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        if response_format:
            payload["response_format"] = response_format
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{_BASE_URL}/openai/chat/completions",
                json=payload,
                headers=self._headers,
            )
            response.raise_for_status()
            data = response.json()
            usage = data.get("usage") or {}
            message = data["choices"][0]["message"]
            return (
                message.get("content"),
                _openai_usage_details(usage),
                message.get("reasoning_content"),
                message.get("tool_calls"),
            )

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
