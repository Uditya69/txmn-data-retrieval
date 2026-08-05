from retrieval_api.gateway_client import GatewayClient

_ELBOW_RATIO = 0.6
_MIN_CHUNKS = 1
_MAX_CHUNKS = 5


def _elbow_cutoff(scores_desc: list[float]) -> int:
    """How many top-scored chunks to keep: extend past the minimum while
    each next score stays within _ELBOW_RATIO of the previous one, so a
    single dominant match yields 1 chunk and a flat distribution yields
    up to _MAX_CHUNKS rather than a fixed count either way."""
    count = min(_MIN_CHUNKS, len(scores_desc))
    for i in range(count, min(len(scores_desc), _MAX_CHUNKS)):
        if scores_desc[i] < _ELBOW_RATIO * scores_desc[i - 1]:
            break
        count += 1
    return count


async def rerank_top_chunks(
    gateway: GatewayClient, query: str, candidates: list[dict], top_n: int | None = None
) -> list[dict]:
    scores = await gateway.rerank(role="reranker", query=query, documents=[c["text"] for c in candidates])
    scored = [{**c, "rerank_score": score} for c, score in zip(candidates, scores)]
    scored.sort(key=lambda row: row["rerank_score"], reverse=True)
    cutoff = top_n if top_n is not None else _elbow_cutoff([row["rerank_score"] for row in scored])
    return scored[:cutoff]
