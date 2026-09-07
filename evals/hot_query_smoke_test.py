"""Hot-query smoke test - our index vs prod, "do we get results at all".

Not a relevance eval (see retrieval_eval.py / retrieval_cases.json for that) - this
only answers "for Taxmann's real top hot search queries, does our ES index
(settings.es_index) return any hits, and does prod's live research API
(research/getSearchResult) return any hits". Pass/fail is purely hit-count > 0.

Query source: misc/Hot Queries/hot_search_queries.csv (query,count - already sorted
count desc, ~8M rows / 310MB). Only the first --top-n rows are ever read off disk
(itertools.islice over a streaming csv.reader) - the file is never loaded whole.

Our index: common.es_client.raw_search against settings.es_index (the same ES this
repo's Instant/AI Mode query - see CLAUDE.md, `researchindex_aic_test` in .env today).
Milvus is deliberately not included here - hybrid_search needs a live model-gateway
(query_embed role -> Voyage) which CLAUDE.md notes is often unreachable in eval
environments; ES alone is what this smoke test needs to answer the hit/no-hit question.

Prod: POST to PROD_SEARCH_URL (default https://liveresearchapi.taxmann.com/api/research/getSearchResult),
body shaped like TaxmannAPI's FilterParamGlobalSearch (Controllers/ResearchElastic/
GlobalSearchIndexController.cs, Models/SearchPropertiesResearch.cs) - only
filter.SearchText is required. Auth is NOT a standard `Authorization: Bearer` header -
verified 2026-09-07 against a real logged-in session's own request: the token goes in
a `refreshtoken` header, value used exactly as captured (its own "Bearer<jwt>" string,
no space, sent as-is - don't reformat it). The site also gates on `appid`, `machineid`,
`origin`/`referer` matching taxmann.com. Read from env vars PROD_API_TOKEN and
PROD_MACHINE_ID (put both in .env, never hardcode or pass on the command line where
they'd land in shell history). Missing/rejected token -> that query's prod_status is
"auth_error", not a crash; the run keeps going so the our-index half of the report
still comes out even with no prod access at all.

Usage (from repo root):
    uv run python evals/hot_query_smoke_test.py --top-n 100

Writes evals/hot_query_smoke_test_results.csv and .json.
"""
import argparse
import asyncio
import csv
import itertools
import json
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

from common.config import get_settings
from common.es_client import get_es_client, raw_search

EVALS_DIR = Path(__file__).parent
DEFAULT_HOT_QUERIES_CSV = EVALS_DIR.parent.parent / "misc" / "Hot Queries" / "hot_search_queries.csv"
DEFAULT_PROD_URL = "https://liveresearchapi.taxmann.com/api/research/getSearchResult"
DEFAULT_OUR_LIMIT = 10
DEFAULT_PROD_PAGE_SIZE = 10
PROD_REQUEST_DELAY_SECONDS = 0.3  # be polite to prod's own ThrottlingHandler


def load_hot_queries(csv_path: Path, top_n: int) -> list[dict]:
    """Streams the first top_n data rows off disk - the file is sorted count-desc
    already (verified: first row count=15294, last row count=1), so "first N after
    the header" is exactly "top N hot queries". Never reads the whole 310MB file."""
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(itertools.islice(reader, top_n))
    for row in rows:
        row["count"] = int(row["count"])
    return rows


async def check_our_index(es_client, query: str, limit: int) -> dict:
    try:
        hits = await raw_search(es_client, query, limit=limit, boost=True)
    except Exception as exception:
        return {"status": "error", "hit_count": 0, "error": f"{type(exception).__name__}: {exception}"}
    return {
        "status": "ok" if hits else "no_results",
        "hit_count": len(hits),
        "top5_doc_ids": [hit["doc_id"] for hit in hits[:5]],  # serial order - rank 1 first
    }


async def check_prod(
    http_client: httpx.AsyncClient, url: str, token: str | None, machine_id: str | None,
    query: str, page_size: int,
) -> dict:
    if not token:
        return {"status": "auth_error", "hit_count": 0, "error": "PROD_API_TOKEN not set"}
    body = {
        "page": 1,
        "pageSize": page_size,
        "filter": {"SearchText": query},
        "sortby": "relevance",
        "sortorder": 1,
        "isUroIncluded": True,
        "isheadnoteToggle": False,
    }
    # Captured verbatim from a working logged-in browser session (2026-09-07) - refreshtoken
    # takes the token exactly as-is (its own "Bearer<jwt>" string, no reformatting), and the
    # site gates on appid/machineid/origin/referer matching a real taxmann.com session.
    headers = {
        "accept": "application/json, text/plain, */*",
        "content-type": "application/json",
        "appid": "2020",
        "origin": "https://www.taxmann.com",
        "referer": "https://www.taxmann.com/",
        "refreshtoken": token,
    }
    if machine_id:
        headers["machineid"] = machine_id
    try:
        response = await http_client.post(url, json=body, headers=headers, timeout=30.0)
    except Exception as exception:
        return {"status": "error", "hit_count": 0, "error": f"{type(exception).__name__}: {exception}"}
    if response.status_code in (401, 403):
        return {"status": "auth_error", "hit_count": 0, "error": f"HTTP {response.status_code}"}
    if response.status_code != 200:
        return {"status": "http_error", "hit_count": 0, "error": f"HTTP {response.status_code}: {response.text[:200]}"}
    try:
        payload = response.json()
    except Exception as exception:
        return {"status": "error", "hit_count": 0, "error": f"bad JSON: {exception}"}
    data = payload.get("Data") or payload.get("data")
    total = (data or {}).get("Total", 0) if isinstance(data, dict) else 0
    result_rows = (data or {}).get("result") or [] if isinstance(data, dict) else []
    top5_doc_ids = [row["Id"] for row in result_rows[:5]]  # GlobalSearchFields.Id - serial order
    return {"status": "ok" if total else "no_results", "hit_count": total, "top5_doc_ids": top5_doc_ids}


async def run(args: argparse.Namespace) -> list[dict]:
    load_dotenv()
    settings = get_settings()
    token = os.environ.get("PROD_API_TOKEN")
    machine_id = os.environ.get("PROD_MACHINE_ID")
    if not token:
        print("WARNING: PROD_API_TOKEN not set - every prod check will be logged as auth_error.")

    queries = load_hot_queries(Path(args.hot_queries_csv), args.top_n)
    print(f"Loaded {len(queries)} hot queries from {args.hot_queries_csv}")

    es_client = get_es_client(settings)
    results: list[dict] = []
    try:
        async with httpx.AsyncClient() as http_client:
            for i, row in enumerate(queries, start=1):
                query = row["query"]
                our = await check_our_index(es_client, query, args.our_limit)
                prod = await check_prod(http_client, args.prod_url, token, machine_id, query, args.prod_page_size)
                if token:
                    await asyncio.sleep(PROD_REQUEST_DELAY_SECONDS)
                result = {
                    "query": query,
                    "hot_count": row["count"],
                    "our_status": our["status"],
                    "our_hit_count": our["hit_count"],
                    # joined with "|" so it round-trips cleanly through csv.DictWriter too
                    "our_top5_doc_ids": "|".join(our.get("top5_doc_ids") or []),
                    "our_error": our.get("error"),
                    "prod_status": prod["status"],
                    "prod_hit_count": prod["hit_count"],
                    "prod_top5_doc_ids": "|".join(prod.get("top5_doc_ids") or []),
                    "prod_error": prod.get("error"),
                }
                results.append(result)
                print(f"[{i}/{len(queries)}] {query!r} -> our={our['status']}({our['hit_count']}) "
                      f"prod={prod['status']}({prod['hit_count']})")
    finally:
        await es_client.close()
    return results


def write_outputs(results: list[dict], out_dir: Path) -> None:
    csv_path = out_dir / "hot_query_smoke_test_results.csv"
    json_path = out_dir / "hot_query_smoke_test_results.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")


def print_summary(results: list[dict]) -> None:
    total = len(results)
    our_zero = [r for r in results if r["our_status"] == "no_results"]
    our_error = [r for r in results if r["our_status"] == "error"]
    prod_zero = [r for r in results if r["prod_status"] == "no_results"]
    prod_auth_error = [r for r in results if r["prod_status"] == "auth_error"]
    prod_ok = [r for r in results if r["prod_status"] == "ok"]

    print("\n=== Summary ===")
    print(f"Total queries checked: {total}")
    print(f"Our index: {total - len(our_zero) - len(our_error)}/{total} returned hits, "
          f"{len(our_zero)} zero-hit, {len(our_error)} errored")
    if prod_auth_error:
        print(f"Prod: {len(prod_auth_error)}/{total} auth_error (no valid PROD_API_TOKEN) - "
              "prod comparison not meaningful for this run")
    print(f"Prod: {len(prod_ok)}/{total} returned hits, {len(prod_zero)} zero-hit "
          f"(out of {total - len(prod_auth_error)} actually reachable)")

    if our_zero:
        print("\nQueries with ZERO hits on our index (highest hot_count first):")
        for r in sorted(our_zero, key=lambda r: -r["hot_count"])[:20]:
            print(f"  [{r['hot_count']}] {r['query']!r}")

    both_zero = [r for r in results if r["our_status"] == "no_results" and r["prod_status"] == "no_results"]
    our_zero_prod_ok = [r for r in results if r["our_status"] == "no_results" and r["prod_status"] == "ok"]
    if our_zero_prod_ok:
        print(f"\nQueries where PROD has results but OUR index doesn't ({len(our_zero_prod_ok)}):")
        for r in sorted(our_zero_prod_ok, key=lambda r: -r["hot_count"])[:20]:
            print(f"  [{r['hot_count']}] {r['query']!r} (prod hits={r['prod_hit_count']})")
    if both_zero:
        print(f"\n{len(both_zero)} queries have zero hits on BOTH - likely a real content gap, not an indexing bug.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hot-queries-csv", default=str(DEFAULT_HOT_QUERIES_CSV))
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--our-limit", type=int, default=DEFAULT_OUR_LIMIT)
    parser.add_argument("--prod-url", default=DEFAULT_PROD_URL)
    parser.add_argument("--prod-page-size", type=int, default=DEFAULT_PROD_PAGE_SIZE)
    parser.add_argument("--output-dir", default=str(EVALS_DIR))
    args = parser.parse_args()

    start = time.monotonic()
    results = asyncio.run(run(args))
    write_outputs(results, Path(args.output_dir))
    print_summary(results)
    print(f"\nDone in {time.monotonic() - start:.1f}s")


if __name__ == "__main__":
    main()
