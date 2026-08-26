"""Ultra-short keyword-only probe: ES BM25 exact-match vs Milvus dense.

Motivating hypothesis (per project owner): a bare 1-3 word statutory-reference
query like "section 55" or "rule 6" is exactly what BM25 nails - the query terms
are a literal substring/heading match - but may be too semantically thin for a
dense embedding to place near the right doc among thousands of near-duplicate
"Section - N" / "Rule - N" chunks.

Because these queries are so generic, there is no single uniquely-correct
doc_id - many docs share the literal heading (e.g. "Section - 54" appears
verbatim across multiple statutes/editions), and ES ties them within noise on
score. So "gold" here is defined as the top-3 ES-ranked doc_ids for that exact
query (near-tied by score, confirmed live against ES on 2026-08-25) - any of
those three counts as a correct answer for the Milvus-side check.

Two things happen when this is run:
  1. Builds/refreshes evals/keyword_only_cases.json from live ES (source of truth).
  2. Immediately runs Milvus dense-only search (query_embed -> Voyage, straight
     dense_vector search, no ES/sparse/rerank/RRF) against MILVUS_COLLECTIONS and
     records the best rank among the 3 gold doc_ids.

Usage:
    uv run python evals/keyword_only_probe.py --gateway-url http://localhost:8001
"""
import argparse
import asyncio
import csv
import json
from pathlib import Path

from common.config import get_settings
from common.es_client import get_es_client, raw_search
from common.milvus_client import get_milvus_client, hybrid_search
from common.schemas import MILVUS_COLLECTIONS
from retrieval_api.ai_mode.retrieve import _flatten
from retrieval_api.gateway_client import GatewayClient
from retrieval_api.retrieval_eval import doc_rank

EVALS_DIR = Path(__file__).parent
LIMIT = 20

# Strict 1-3 word keyword-only queries - no case names, no descriptive text,
# just the bare statutory reference, covering acts/rules/articles and both
# "section N" and bare-code ("80HH") phrasing.
KEYWORD_QUERIES = [
    "section 54",
    "section 55",
    "section 43B",
    "section 61",
    "section 271",
    "rule 6",
    "article 14",
    "80HH",
]


async def build_cases(es_client) -> list[dict]:
    cases = []
    for index, query in enumerate(KEYWORD_QUERIES, start=1):
        hits = await raw_search(es_client, query, limit=3)
        if not hits:
            continue
        cases.append({
            "id": f"K{index:02d}",
            "class": "keyword",
            "query": query,
            "gold_doc_ids": [h["doc_id"] for h in hits],
            "es_top_headings": [h["heading"] for h in hits],
            "es_top_scores": [h["score"] for h in hits],
            "pass_at": 5,
        })
    return cases


async def run_case(gateway: GatewayClient, milvus_client, case: dict, limit: int) -> dict:
    query = case["query"]
    gold = set(case["gold_doc_ids"])
    vector = await gateway.embed(role="query_embed", text=query)
    by_collection = await hybrid_search(milvus_client, MILVUS_COLLECTIONS, vector, query, limit=limit)
    flat = _flatten(by_collection, "milvus_dense")
    rank = doc_rank(flat, gold)
    top_hits = [
        {"rank": i, "doc_id": row["doc_id"], "collection": row["collection"], "score": row["score"]}
        for i, row in enumerate(flat[:10], start=1)
    ]
    return {
        **case,
        "milvus_dense_rank": rank,
        "hit": rank is not None and rank <= case["pass_at"],
        "milvus_top_hits": top_hits,
    }


def write_csv(results: list[dict], path: Path) -> None:
    fields = [
        "id", "query", "es_top1_doc_id", "es_top1_heading", "gold_doc_ids",
        "milvus_dense_rank", "hit", "milvus_top1_doc_id", "milvus_top1_score",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in results:
            top1 = r["milvus_top_hits"][0] if r["milvus_top_hits"] else None
            writer.writerow({
                "id": r["id"],
                "query": r["query"],
                "es_top1_doc_id": r["gold_doc_ids"][0],
                "es_top1_heading": r["es_top_headings"][0],
                "gold_doc_ids": ";".join(r["gold_doc_ids"]),
                "milvus_dense_rank": r["milvus_dense_rank"] if r["milvus_dense_rank"] is not None else f">{LIMIT}",
                "hit": "Y" if r["hit"] else "N",
                "milvus_top1_doc_id": top1["doc_id"] if top1 else "",
                "milvus_top1_score": top1["score"] if top1 else "",
            })


async def _run(args) -> None:
    settings = get_settings()
    es_client = get_es_client(settings)
    milvus_client = get_milvus_client(settings)
    gateway = GatewayClient(args.gateway_url or settings.gateway_url, trace_enabled=False)
    try:
        cases = await build_cases(es_client)
        (EVALS_DIR / "keyword_only_cases.json").write_text(json.dumps(cases, indent=2))
        print(f"Built {len(cases)} keyword-only cases from live ES -> evals/keyword_only_cases.json")

        results = []
        for case in cases:
            result = await run_case(gateway, milvus_client, case, LIMIT)
            results.append(result)
            print(f"{result['id']} {result['query']!r}: ES top1={result['gold_doc_ids'][0]} ({result['es_top_headings'][0]!r}) | milvus_dense_rank={result['milvus_dense_rank'] or f'>{LIMIT}'} hit={result['hit']}")
    finally:
        await es_client.close()
        milvus_client.close()

    json_path = EVALS_DIR / "keyword_only_results.json"
    csv_path = EVALS_DIR / "keyword_only_results.csv"
    json_path.write_text(json.dumps(results, indent=2))
    write_csv(results, csv_path)

    hits = sum(r["hit"] for r in results)
    print()
    print(f"Milvus dense recall@5 on gold set: {hits}/{len(results)}")
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ultra-short keyword-only ES-vs-Milvus-dense probe")
    parser.add_argument("--gateway-url", help="override GATEWAY_URL (use http://localhost:8001 when running from the host)")
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
