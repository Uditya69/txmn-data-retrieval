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
    # An empty candidates list (e.g. ES+Milvus both return zero hits for an obscure/gibberish
    # query) means an empty `documents` list here - DeepInfra's rerank endpoint 422s on that
    # ("the number of queries and documents must be the same"), which model-gateway doesn't
    # catch and turns into an unhandled 500 all the way back to the client. Nothing to rerank
    # anyway, so skip the call entirely - same guard instant/rerank.py already has.
    if not candidates:
        return []
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
