import httpx

# Self-hosted OpenAI-compatible chat endpoint (e.g. vLLM serving qwen3).
# Chat-only: no embed/rerank routes exist on this server, so roles routed
# here must never be "query_embed" or "reranker".
_CHAT_MAX_TOKENS = 32768
# "synthesis" gets a much larger completion budget than the default above. qwen3's hybrid
# thinking mode spends this budget on <think> chain-of-thought before ever writing the answer,
# and a synthesis prompt (system prompt + up to 5 full excerpts) reasons far longer than the
# trivial prompts 32768 was sized against - observed live: the budget exhausted mid-reasoning,
# cutting the response off by length with no finished answer at all, not a timeout (the outer
# 600s/620s timeouts had ample room left). The model actually served here (qwen3, see
# .env LOCAL_CHAT_MODEL_SYNTHESIS) has a 262144-token native context, so 32768 was leaving most
# of that budget unused rather than reflecting any real limit - raised to stay comfortably
# under context while giving reasoning + answer enough room to both complete normally.
#
# "slm" (intent extraction, extract_intent) hit the identical failure mode: observed live via
# Langfuse, a reasoning trace cut off mid-thought with no closing JSON, which json.loads then
# rejects, silently triggering intent.py's _fallback_intent (search_query passed through
# unchanged, intent: []) - indistinguishable from the model genuinely declining to expand
# unless you go read the reasoning trace and notice it never finished. temperature=0.6 (see
# extract_intent's call site) makes the Thinking model's reasoning length variable run to run,
# so this wasn't reliably triggered by any specific query. Given the same 131072 budget as
# synthesis rather than a smaller one, since intent's reasoning length is driven by the same
# Thinking-mode chain-of-thought behavior, not by prompt size - no evidence the two roles'
# worst-case reasoning length differs enough to size them apart.
_CHAT_MAX_TOKENS_BY_ROLE = {"synthesis": 131072, "slm": 131072}


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
        self, model: str, messages: list[dict], response_format: dict | None = None,
        temperature: float | None = None, role: str | None = None, reasoning_enabled: bool = True,
    ) -> tuple[str | None, dict[str, int], str | None]:
        max_tokens = _CHAT_MAX_TOKENS_BY_ROLE.get(role, _CHAT_MAX_TOKENS)
        payload = {"model": model, "messages": messages, "max_tokens": max_tokens}
        if response_format:
            payload["response_format"] = response_format
        if temperature is not None:
            payload["temperature"] = temperature
        if not reasoning_enabled:
            # vLLM's standard toggle for Qwen3's hybrid thinking mode - skips the
            # <think> chain-of-thought entirely rather than just hiding it, so this
            # also sidesteps the max_tokens-exhausted-mid-reasoning failure mode
            # documented above, not just its latency/token cost.
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        # Self-hosted qwen3 observed taking 30-70s even on a trivial 2-line prompt
        # (verified via Postman: 69.4s) - AI Mode's synthesis prompt is far larger, so
        # the old 60s timeout here raced the model's own response time and lost.
        # Self-hosted, no rate-limit/cost pressure to bound this tightly like DeepInfra -
        # set high (10 min) so slow generations aren't the failure mode; a genuinely dead
        # server is still caught, just later. gateway_client.py's outer timeout must
        # stay >= this.
        async with httpx.AsyncClient(timeout=600.0) as client:
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
            )
