from retrieval_api.gateway_client import GatewayClient


async def rerank_top_chunks(
    gateway: GatewayClient, query: str, candidates: list[dict], top_n: int = 3
) -> list[dict]:
    scores = await gateway.rerank(role="reranker", query=query, documents=[c["text"] for c in candidates])
    scored = [{**c, "rerank_score": score} for c, score in zip(candidates, scores)]
    scored.sort(key=lambda row: row["rerank_score"], reverse=True)
    return scored[:top_n]
