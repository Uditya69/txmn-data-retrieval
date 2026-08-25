import asyncio
import logging

from fastapi import APIRouter
from pydantic import BaseModel

from common.config import get_settings
from common.es_client import get_es_client, raw_search
from common.milvus_client import get_milvus_client, hybrid_search
from common.schemas import MILVUS_COLLECTIONS, collections_for_intent
from retrieval_api.ai_mode.rerank import rerank_top_chunks
from retrieval_api.gateway_client import GatewayClient
from retrieval_api.instant.rerank import rrf_merge_by_doc_id
from retrieval_api.trace_utils import collection_trace

router = APIRouter()

logger = logging.getLogger(__name__)

_ES_LIMIT = 20  # matches instant/search.py's own ES call size
_MILVUS_LIMIT = 50  # matches instant/search.py and ai_mode/retrieve.py's own per-collection size

# Same cap ai_mode/citations.py::rerank_and_prefetch uses before its DeepInfra rerank call -
# a fused ES+Milvus-dense candidate list is unbounded (up to 11 collections x 50 each), and
# sending every one of those texts to the reranker in a single request risks the same
# request-size/timeout failure mode documented there.
_MAX_RERANK_CANDIDATES = 100


def _flatten_dense(by_collection: dict[str, list[dict]]) -> list[dict]:
    rows = [row for rows in by_collection.values() for row in rows]
    return sorted(rows, key=lambda row: row["score"], reverse=True)


class KeywordSearchRequest(BaseModel):
    query: str
    boost: bool = False
    limit: int = _ES_LIMIT


@router.post("/v1/search/keyword")
async def keyword_only_search(req: KeywordSearchRequest):
    """ES lexical search only - no Milvus call at all, no RRF, no reranker. The forced
    version of what the instant_classifier's KEYWORD label routes to (routing_plan:
    {"es": True, "milvus": False}) - always runs this path regardless of what the
    classifier would have picked, for isolating ES's own retrieval quality."""
    settings = get_settings()
    es_client = get_es_client(settings)
    try:
        hits = await raw_search(es_client, req.query, limit=req.limit, boost=req.boost)
        return {"query": req.query, "hit_count": len(hits), "hits": hits}
    finally:
        await es_client.close()


class IntentSearchRequest(BaseModel):
    query: str
    intent: list[str] = []
    limit: int = _MILVUS_LIMIT


@router.post("/v1/search/intent")
async def intent_only_search(req: IntentSearchRequest):
    """Milvus dense (semantic) search only - no ES, no Milvus sparse, no fusion. The forced
    version of what the instant_classifier's INTENT label routes to (routing_plan:
    {"es": False, "milvus": True}). `intent` optionally routes to a category-specific
    collection subset via collections_for_intent(); omitted/empty searches all 11
    collections, matching AI Mode's own no-intent fallback."""
    settings = get_settings()
    gateway = GatewayClient(base_url=settings.gateway_url)
    try:
        milvus_client = get_milvus_client(settings)
    except Exception:
        logger.exception("Milvus connection failed")
        return {"query": req.query, "error": "milvus_unavailable", "collections": []}

    try:
        collections = collections_for_intent(req.intent) if req.intent else MILVUS_COLLECTIONS
        dense_vector = await gateway.embed(role="query_embed", text=req.query)
        by_collection = await hybrid_search(
            milvus_client, collections=collections, dense_vector=dense_vector,
            sparse_query_text=req.query, limit=req.limit,
        )
        return {"query": req.query, **collection_trace(by_collection)}
    finally:
        milvus_client.close()


class HybridSearchRequest(BaseModel):
    query: str
    intent: list[str] = []
    boost: bool = False
    limit: int = _MILVUS_LIMIT
    top_n: int | None = None  # None -> rerank_top_chunks' own elbow_cutoff (max 5)


@router.post("/v1/search/hybrid")
async def hybrid_search_with_rerank(req: HybridSearchRequest):
    """ES lexical + Milvus dense, RRF-fused by rank position (never raw score - CLAUDE.md
    hard rule 3), then the real DeepInfra cross-encoder reranker on top - the forced version
    of what the instant_classifier's HYBRID label routes to, but with an actual reranker
    stage layered on (Instant mode's own "rerank" step is RRF only, no cross-encoder call).
    No Milvus sparse pass at all - dense is the only Milvus signal here."""
    settings = get_settings()
    gateway = GatewayClient(base_url=settings.gateway_url)
    es_client = get_es_client(settings)
    try:
        milvus_client = get_milvus_client(settings)
    except Exception:
        logger.exception("Milvus connection failed")
        milvus_client = None

    try:
        collections = collections_for_intent(req.intent) if req.intent else MILVUS_COLLECTIONS
        dense_vector = await gateway.embed(role="query_embed", text=req.query)

        async def _run_dense():
            if milvus_client is None:
                return {}
            return await hybrid_search(
                milvus_client, collections=collections, dense_vector=dense_vector,
                sparse_query_text=req.query, limit=req.limit,
            )

        es_result, dense_by_collection = await asyncio.gather(
            raw_search(es_client, req.query, limit=_ES_LIMIT, boost=req.boost),
            _run_dense(),
        )

        fused = rrf_merge_by_doc_id(
            {"es": es_result, "milvus_dense": _flatten_dense(dense_by_collection)},
            {"es": 1.0, "milvus_dense": 1.0},
        )
        rerank_candidates = fused[:_MAX_RERANK_CANDIDATES]
        top_chunks = await rerank_top_chunks(gateway, req.query, rerank_candidates, top_n=req.top_n)

        return {
            "query": req.query,
            "candidate_count": len(fused),
            "considered_count": len(rerank_candidates),
            "top_chunks": top_chunks,
        }
    finally:
        await es_client.close()
        if milvus_client is not None:
            milvus_client.close()
