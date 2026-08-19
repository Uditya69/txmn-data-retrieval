import httpx
from langfuse import get_client

# Retrieval-api and model-gateway are separate processes; Langfuse's OTel
# context doesn't cross the HTTP hop on its own, so the current trace/span
# id is forwarded as headers and re-attached on the gateway side via
# trace_context (see model_gateway.routes._trace_context_from_headers).
_TRACE_ID_HEADER = "x-langfuse-trace-id"
_PARENT_SPAN_ID_HEADER = "x-langfuse-parent-observation-id"


def _trace_headers() -> dict[str, str]:
    langfuse = get_client()
    trace_id = langfuse.get_current_trace_id()
    observation_id = langfuse.get_current_observation_id()
    if not trace_id or not observation_id:
        return {}
    return {_TRACE_ID_HEADER: trace_id, _PARENT_SPAN_ID_HEADER: observation_id}


class GatewayClient:
    def __init__(self, base_url: str, trace_enabled: bool = True):
        self._base_url = base_url
        self._trace_enabled = trace_enabled

    def _headers(self) -> dict[str, str]:
        return _trace_headers() if self._trace_enabled else {}

    async def chat(
        self, role: str, messages: list[dict], model: str | None = None,
        response_format: dict | None = None, temperature: float | None = None,
    ) -> str:
        content, _reasoning = await self.chat_with_reasoning(
            role, messages, model=model, response_format=response_format, temperature=temperature,
        )
        return content

    async def chat_with_reasoning(
        self, role: str, messages: list[dict], model: str | None = None,
        response_format: dict | None = None, temperature: float | None = None,
    ) -> tuple[str, str | None]:
        body = {"role": role, "messages": messages}
        if model is not None:
            body["model"] = model
        if response_format is not None:
            body["response_format"] = response_format
        if temperature is not None:
            body["temperature"] = temperature
        # Must stay >= local.py's LocalAdapter timeout (600s) - this wraps that call over
        # HTTP, so a shorter outer timeout would cut the request before the inner one
        # ever gets to time out itself, defeating it.
        async with httpx.AsyncClient(timeout=620.0) as client:
            response = await client.post(
                f"{self._base_url}/v1/chat", json=body, headers=self._headers(),
            )
            response.raise_for_status()
            data = response.json()
            return data["content"], data.get("reasoning")

    async def chat_with_tools(
        self, role: str, messages: list[dict], tools: list[dict], tool_choice: str | None = None,
    ) -> dict:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self._base_url}/v1/chat",
                json={"role": role, "messages": messages, "tools": tools, "tool_choice": tool_choice},
                headers=_trace_headers(),
            )
            response.raise_for_status()
            data = response.json()
            return {"content": data.get("content"), "tool_calls": data.get("tool_calls"), "reasoning": data.get("reasoning")}

    async def get_model(self, role: str) -> str:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(f"{self._base_url}/v1/models/{role}")
            response.raise_for_status()
            return response.json()["model"]

    async def embed(self, role: str, text: str) -> list[float]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self._base_url}/v1/embed", json={"role": role, "text": text}, headers=self._headers(),
            )
            response.raise_for_status()
            return response.json()["embedding"]

    async def rerank(
        self, role: str, query: str, documents: list[str], model: str | None = None,
    ) -> list[float]:
        body = {"role": role, "query": query, "documents": documents}
        if model is not None:
            body["model"] = model
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self._base_url}/v1/rerank",
                json=body,
                headers=self._headers(),
            )
            response.raise_for_status()
            return response.json()["scores"]
