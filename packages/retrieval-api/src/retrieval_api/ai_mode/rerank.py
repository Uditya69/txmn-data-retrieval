from retrieval_api.gateway_client import GatewayClient
from retrieval_api.score_cutoff import elbow_cutoff

_MAX_CHUNKS = 5

# retrieve()'s RRF-merged candidates can run into the hundreds (up to 50 hits per
# collection x2 dense/sparse x7 collections, deduped by chunk_id). Sending all of that
# as reranker API input on every query is real, avoidable token/latency cost - cap to
# the top-N by rrf_score (the fused rank signal, already ES-boosted) before the call,
# and let elbow_cutoff still trim the final selection from the reranked scores below.
_MAX_RERANK_CANDIDATES = 50


async def rerank_top_chunks(
    gateway: GatewayClient, query: str, candidates: list[dict],
    top_n: int | None = None, model: str | None = None,
) -> list[dict]:
    capped = sorted(candidates, key=lambda row: row["rrf_score"], reverse=True)[:_MAX_RERANK_CANDIDATES]
    scores = await gateway.rerank(
        role="reranker", query=query, documents=[c["text"] for c in capped], model=model,
    )
    scored = [{**c, "rerank_score": score} for c, score in zip(capped, scores)]
    scored.sort(key=lambda row: row["rerank_score"], reverse=True)
    cutoff = top_n if top_n is not None else elbow_cutoff(
        [row["rerank_score"] for row in scored], max_keep=_MAX_CHUNKS,
    )
    return scored[:cutoff]
