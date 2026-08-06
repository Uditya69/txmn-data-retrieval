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

    async def chat(self, role: str, messages: list[dict]) -> str:
        content, _reasoning = await self.chat_with_reasoning(role, messages)
        return content

    async def chat_with_reasoning(self, role: str, messages: list[dict]) -> tuple[str, str | None]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self._base_url}/v1/chat", json={"role": role, "messages": messages}, headers=self._headers(),
            )
            response.raise_for_status()
            data = response.json()
            return data["content"], data.get("reasoning")

    async def embed(self, role: str, text: str) -> list[float]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self._base_url}/v1/embed", json={"role": role, "text": text}, headers=self._headers(),
            )
            response.raise_for_status()
            return response.json()["embedding"]

    async def rerank(self, role: str, query: str, documents: list[str]) -> list[float]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self._base_url}/v1/rerank",
                json={"role": role, "query": query, "documents": documents},
                headers=self._headers(),
            )
            response.raise_for_status()
            return response.json()["scores"]
