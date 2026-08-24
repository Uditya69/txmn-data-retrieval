from fastapi import APIRouter
from pydantic import BaseModel

from common.es_client import build_query_preview

router = APIRouter()


class QueryAnalysisRequest(BaseModel):
    query: str
    boost: bool = False


@router.post("/v1/query-analysis")
async def get_query_analysis(req: QueryAnalysisRequest):
    """Our equivalent of centax-node's `/research-premium/api/v1/getLowLevelQuery`: given a
    raw search string, returns exactly how Instant mode's raw_search (common/es_client.py)
    breaks it down and queries ES - shape classification, synonym expansion, chunking
    (query_tokenizer.chunk_query - proximity-phrase groups, our version of centax-node's
    QueryTokens), and the actual ES query body - with no search executed, no ES round trip.
    Exists to let a query's breakdown be compared side by side against centax-node's own
    endpoint on demand, instead of only being visible inside a running search's trace panel.
    `boost` mirrors Instant mode's ranking-boost toggle - default False shows the plain query,
    matching raw_search's own default."""
    return build_query_preview(req.query, boost=req.boost)
