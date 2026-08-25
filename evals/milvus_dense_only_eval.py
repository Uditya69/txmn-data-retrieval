"""Milvus-dense-only retrieval eval.

Embeds each query as-is (query_embed role -> Voyage, per CLAUDE.md hard rule 1) and
searches Milvus's dense_vector field directly across all collections - no ES, no
sparse pass, no RRF fusion, no reranker. Answers one question: with nothing but a
straight embed-and-cosine-search, where does the gold doc_id land?

Pulls cases from evals/retrieval_cases.json (case law) and evals/statutory_cases.json
(acts/rules/articles/commentary), class in {direct, indirect} only (adversarial
excluded - out of scope for this pass). To hit the requested ~60-70% direct mix
(the two source datasets are ~50/50 direct/indirect on their own), all direct cases
are kept and indirect cases are subsampled by id order.

Usage (from repo root, gateway running via docker compose):
    uv run python evals/milvus_dense_only_eval.py --gateway-url http://localhost:8001

Writes evals/milvus_dense_only_results.csv (one row per query) and
evals/milvus_dense_only_results.json (full detail, top-20 hits per query).
"""
import argparse
import asyncio
import csv
import json
from pathlib import Path

from common.config import get_settings
from common.milvus_client import get_milvus_client, hybrid_search
from common.schemas import MILVUS_COLLECTIONS
from retrieval_api.ai_mode.retrieve import _flatten
from retrieval_api.gateway_client import GatewayClient
from retrieval_api.retrieval_eval import doc_rank

EVALS_DIR = Path(__file__).parent
TOP_K_RECORDED = 20
INDIRECT_SAMPLE_STEP = 2  # keep every other indirect case -> ~65% direct overall


def load_mixed_cases() -> list[dict]:
    cases = []
    for dataset, path in (
        ("caselaw", EVALS_DIR / "retrieval_cases.json"),
        ("statutory", EVALS_DIR / "statutory_cases.json"),
    ):
        for case in json.loads(path.read_text()):
            if case["class"] in ("direct", "indirect"):
                cases.append({**case, "source_dataset": dataset})

    direct = [c for c in cases if c["class"] == "direct"]
    indirect = [c for c in cases if c["class"] == "indirect"]
    indirect.sort(key=lambda c: c["id"])
    indirect_sample = indirect[::INDIRECT_SAMPLE_STEP]
    mixed = direct + indirect_sample
    mixed.sort(key=lambda c: (c["source_dataset"], c["id"]))
    return mixed


async def run_case(gateway: GatewayClient, milvus_client, case: dict, limit: int) -> dict:
    query = case["query"]
    gold = set(case["gold_doc_ids"])
    vector = await gateway.embed(role="query_embed", text=query)
    by_collection = await hybrid_search(milvus_client, MILVUS_COLLECTIONS, vector, query, limit=limit)
    flat = _flatten(by_collection, "milvus_dense")
    rank = doc_rank(flat, gold)
    top_hits = [
        {
            "rank": index,
            "doc_id": row["doc_id"],
            "collection": row["collection"],
            "score": row["score"],
            "is_gold": row["doc_id"] in gold,
        }
        for index, row in enumerate(flat[:TOP_K_RECORDED], start=1)
    ]
    return {
        "id": case["id"],
        "source_dataset": case["source_dataset"],
        "class": case["class"],
        "query": query,
        "gold_doc_ids": sorted(gold),
        "expected_collections": case["expected_collections"],
        "pass_at": case["pass_at"],
        "rank": rank,
        "hit": rank is not None and rank <= case["pass_at"],
        "top_hits": top_hits,
    }


def write_csv(results: list[dict], path: Path) -> None:
    fields = [
        "id", "source_dataset", "class", "query", "gold_doc_ids", "expected_collections",
        "pass_at", "rank", "hit",
        "top1_doc_id", "top1_collection", "top1_score",
        "top5_doc_ids",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in results:
            top1 = r["top_hits"][0] if r["top_hits"] else None
            writer.writerow({
                "id": r["id"],
                "source_dataset": r["source_dataset"],
                "class": r["class"],
                "query": r["query"],
                "gold_doc_ids": ";".join(r["gold_doc_ids"]),
                "expected_collections": ";".join(r["expected_collections"]),
                "pass_at": r["pass_at"],
                "rank": r["rank"] if r["rank"] is not None else f">{TOP_K_RECORDED}",
                "hit": "Y" if r["hit"] else "N",
                "top1_doc_id": top1["doc_id"] if top1 else "",
                "top1_collection": top1["collection"] if top1 else "",
                "top1_score": top1["score"] if top1 else "",
                "top5_doc_ids": ";".join(h["doc_id"] for h in r["top_hits"][:5]),
            })


async def _run(args) -> None:
    cases = load_mixed_cases()
    direct_pct = round(100 * sum(c["class"] == "direct" for c in cases) / len(cases), 1)
    print(f"{len(cases)} cases loaded ({direct_pct}% direct)")

    settings = get_settings()
    milvus_client = get_milvus_client(settings)
    gateway = GatewayClient(args.gateway_url or settings.gateway_url, trace_enabled=False)
    try:
        results = []
        for index, case in enumerate(cases, start=1):
            result = await run_case(gateway, milvus_client, case, args.limit)
            results.append(result)
            print(f"[{index}/{len(cases)}] {result['id']} ({result['class']}): rank={result['rank'] or f'>{args.limit}'} hit={result['hit']}")
    finally:
        milvus_client.close()

    json_path = EVALS_DIR / "milvus_dense_only_results.json"
    csv_path = EVALS_DIR / "milvus_dense_only_results.csv"
    json_path.write_text(json.dumps(results, indent=2))
    write_csv(results, csv_path)

    hits = sum(r["hit"] for r in results)
    direct_hits = sum(r["hit"] for r in results if r["class"] == "direct")
    indirect_hits = sum(r["hit"] for r in results if r["class"] == "indirect")
    direct_n = sum(r["class"] == "direct" for r in results)
    indirect_n = sum(r["class"] == "indirect" for r in results)
    print()
    print(f"Overall: {hits}/{len(results)} passed pass_at threshold")
    print(f"Direct:   {direct_hits}/{direct_n}")
    print(f"Indirect: {indirect_hits}/{indirect_n}")
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Milvus dense-only retrieval eval (no ES, no sparse, no rerank)")
    parser.add_argument("--limit", type=int, default=TOP_K_RECORDED, help="Milvus search limit per collection")
    parser.add_argument("--gateway-url", help="override GATEWAY_URL (use http://localhost:8001 when running from the host)")
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
