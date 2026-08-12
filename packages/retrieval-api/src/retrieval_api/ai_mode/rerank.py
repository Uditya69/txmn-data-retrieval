from retrieval_api.gateway_client import GatewayClient
from retrieval_api.score_cutoff import elbow_cutoff

_MAX_CHUNKS = 5


async def rerank_top_chunks(
    gateway: GatewayClient, query: str, candidates: list[dict],
    top_n: int | None = None, model: str | None = None,
) -> list[dict]:
    scores = await gateway.rerank(
        role="reranker", query=query, documents=[c["text"] for c in candidates], model=model,
    )
    scored = [{**c, "rerank_score": score} for c, score in zip(candidates, scores)]
    scored.sort(key=lambda row: row["rerank_score"], reverse=True)
    cutoff = top_n if top_n is not None else elbow_cutoff(
        [row["rerank_score"] for row in scored], max_keep=_MAX_CHUNKS,
    )
    return scored[:cutoff]
