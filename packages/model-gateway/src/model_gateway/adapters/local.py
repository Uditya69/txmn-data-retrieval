import httpx

# Self-hosted OpenAI-compatible chat endpoint (e.g. vLLM serving qwen3).
# Chat-only: no embed/rerank routes exist on this server, so roles routed
# here must never be "query_embed" or "reranker".
_CHAT_MAX_TOKENS = 32768


def _openai_usage_details(usage: dict) -> dict[str, int]:
    """Map an OpenAI-shaped usage block to Langfuse's usage_details keys."""
    details = {}
    if "prompt_tokens" in usage:
        details["input"] = usage["prompt_tokens"]
    if "completion_tokens" in usage:
        details["output"] = usage["completion_tokens"]
    return details


class LocalAdapter:
    def __init__(self, base_url: str, api_key: str):
        self._base_url = base_url.rstrip("/")
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
                f"{self._base_url}/chat/completions",
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
