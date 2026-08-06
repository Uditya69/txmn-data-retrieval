import argparse
import asyncio
import hashlib
import json
import os
import re
import subprocess
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

from langfuse import get_client

from common.config import get_settings
from common.es_client import get_es_client, raw_search
from common.milvus_client import get_milvus_client, hybrid_search
from common.schemas import MILVUS_COLLECTIONS
from retrieval_api.ai_mode.filter_resolve import resolve_allowlist
from retrieval_api.ai_mode.intent import extract_intent
from retrieval_api.ai_mode.rerank import rerank_top_chunks
from retrieval_api.ai_mode.retrieve import _flatten, rrf_merge
from retrieval_api.gateway_client import GatewayClient


def load_cases(path: str | Path) -> list[dict]:
    cases = json.loads(Path(path).read_text())
    if not isinstance(cases, list) or not cases:
        raise ValueError("eval dataset must be a non-empty JSON array")
    seen: set[str] = set()
    for case in cases:
        required = {"id", "class", "query", "gold_doc_ids", "expected_collections", "pass_at"}
        missing = required - case.keys()
        if missing:
            raise ValueError(f"{case.get('id', '<unknown>')}: missing {sorted(missing)}")
        if case["id"] in seen:
            raise ValueError(f"duplicate query id: {case['id']}")
        seen.add(case["id"])
        unknown = set(case["expected_collections"]) - set(MILVUS_COLLECTIONS)
        if unknown:
            raise ValueError(f"{case['id']}: unknown collection(s): {sorted(unknown)}")
        if not case["query"] or not case["gold_doc_ids"]:
            raise ValueError(f"{case['id']}: query and gold_doc_ids must be non-empty")
    return cases


def doc_rank(rows: list[dict], gold_doc_ids: set[str]) -> int | None:
    seen: set[str] = set()
    rank = 0
    for row in rows:
        doc_id = str(row["doc_id"])
        if doc_id in seen:
            continue
        seen.add(doc_id)
        rank += 1
        if doc_id in gold_doc_ids:
            return rank
    return None


def _run_paths(output: Path | None, run_name: str, created_at: datetime) -> tuple[Path, Path, Path | None, Path | None]:
    if output is not None:
        return output, output.with_suffix(".dataset.json"), None, None
    timestamp = created_at.strftime("%Y%m%dT%H%M%S%fZ")
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", run_name).strip("-") or "retrieval-eval"
    base = Path(".eval-results") / f"{timestamp}-{slug}"
    return base.with_suffix(".json"), base.with_suffix(".dataset.json"), Path(".eval-results/latest.json"), Path(".eval-results/latest.dataset.json")


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_dirty() -> bool | None:
    try:
        return bool(subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True,
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def _collection_ranks(by_collection: dict[str, list[dict]], gold: set[str]) -> dict[str, int | None]:
    return {name: doc_rank(rows, gold) for name, rows in by_collection.items()}


async def evaluate_case(case: dict, gateway, es_client, milvus_client, *, limit: int = 50,
                        langfuse_enabled: bool = True) -> dict:
    query = case["query"]
    gold = set(case["gold_doc_ids"])
    langfuse = get_client()
    context = langfuse.start_as_current_observation(
        as_type="evaluator", name="retrieval-eval", input={"id": case["id"], "query": query},
        metadata={"class": case["class"], "pair": case.get("pair", "")},
    ) if langfuse_enabled else nullcontext(None)

    with context as span:
        started = time.perf_counter()
        errors: dict[str, str] = {}
        timings: dict[str, float] = {}

        async def measured(name, awaitable):
            stage_started = time.perf_counter()
            try:
                return await awaitable
            except Exception as exc:  # keep independent diagnostic stages visible
                errors[name] = f"{type(exc).__name__}: {exc}"
                return None
            finally:
                timings[name] = round((time.perf_counter() - stage_started) * 1000, 1)

        es_rows = await measured("es", raw_search(es_client, query, limit=limit)) or []
        raw_vector = await measured("raw_embedding", gateway.embed(role="query_embed", text=query))
        raw_dense = (
            await measured("raw_dense", hybrid_search(
                milvus_client, MILVUS_COLLECTIONS, raw_vector, query, limit=limit,
            )) if raw_vector is not None else None
        ) or {name: [] for name in MILVUS_COLLECTIONS}
        raw_sparse = await measured("raw_sparse", hybrid_search(
            milvus_client, MILVUS_COLLECTIONS, None, query, limit=limit,
        )) or {name: [] for name in MILVUS_COLLECTIONS}

        intent = await measured("intent", extract_intent(gateway, query))
        rewritten_query = intent.get("rewritten_query", query) if intent else query
        allowlist = await measured("filters", resolve_allowlist(es_client, intent.get("filters", {}))) if intent else None
        rewritten_vector = raw_vector if rewritten_query == query else await measured(
            "rewritten_embedding", gateway.embed(role="query_embed", text=rewritten_query),
        )
        rewritten_dense = (
            await measured("rewritten_dense", hybrid_search(
                milvus_client, MILVUS_COLLECTIONS, rewritten_vector, rewritten_query,
                doc_id_allowlist=allowlist, limit=limit,
            )) if rewritten_vector is not None else None
        ) or {name: [] for name in MILVUS_COLLECTIONS}
        rewritten_sparse = await measured("rewritten_sparse", hybrid_search(
            milvus_client, MILVUS_COLLECTIONS, None, rewritten_query,
            doc_id_allowlist=allowlist, limit=limit,
        )) or {name: [] for name in MILVUS_COLLECTIONS}

        dense_flat = _flatten(rewritten_dense)
        sparse_flat = _flatten(rewritten_sparse)
        merged = rrf_merge(dense_flat, sparse_flat)
        reranked = await measured(
            "reranker", rerank_top_chunks(gateway, query, merged, top_n=len(merged)),
        ) if merged else []
        reranked = reranked or []

        ranks = {
            "es": doc_rank(es_rows, gold),
            "raw_dense": doc_rank(_flatten(raw_dense), gold),
            "raw_sparse": doc_rank(_flatten(raw_sparse), gold),
            "rewritten_dense": doc_rank(dense_flat, gold),
            "rewritten_sparse": doc_rank(sparse_flat, gold),
            "rrf": doc_rank(merged, gold),
            "reranker": doc_rank(reranked, gold),
        }
        result = {
            "id": case["id"], "pair": case.get("pair"), "class": case["class"],
            "query": query, "gold_doc_ids": case["gold_doc_ids"],
            "rewritten_query": rewritten_query, "pass_at": case["pass_at"],
            "ranks": ranks,
            "collection_ranks": {
                "raw_dense": _collection_ranks(raw_dense, gold),
                "raw_sparse": _collection_ranks(raw_sparse, gold),
                "rewritten_dense": _collection_ranks(rewritten_dense, gold),
                "rewritten_sparse": _collection_ranks(rewritten_sparse, gold),
            },
            "errors": errors, "timings_ms": timings,
            "total_ms": round((time.perf_counter() - started) * 1000, 1),
        }
        if langfuse_enabled:
            span.update(output={"ranks": ranks, "errors": errors, "rewritten_query": rewritten_query})
            for stage, rank in ranks.items():
                langfuse.score_current_trace(
                    name=f"{stage}_recall_at_{case['pass_at']}", value=rank is not None and rank <= case["pass_at"],
                    data_type="BOOLEAN",
                )
                if rank is not None:
                    langfuse.score_current_trace(name=f"{stage}_rank", value=float(rank), data_type="NUMERIC")
        return result


def _print_summary(results: list[dict]) -> None:
    stages = ["es", "raw_dense", "raw_sparse", "rewritten_dense", "rewritten_sparse", "rrf", "reranker"]
    print("ID   class     " + "  ".join(f"{s[:8]:>8}" for s in stages))
    for result in results:
        values = [str(result["ranks"][stage] or ">50") for stage in stages]
        print(f"{result['id']:<4} {result['class']:<9} " + "  ".join(f"{v:>8}" for v in values))
    print()
    for stage in stages:
        passed = sum(
            result["ranks"][stage] is not None and result["ranks"][stage] <= result["pass_at"]
            for result in results
        )
        print(f"{stage}: {passed}/{len(results)} passed query threshold")


async def _run(args) -> int:
    if args.langfuse_base_url:
        os.environ["LANGFUSE_BASE_URL"] = args.langfuse_base_url
    cases = load_cases(args.dataset)
    if args.query:
        wanted = set(args.query)
        cases = [case for case in cases if case["id"] in wanted]
        missing = wanted - {case["id"] for case in cases}
        if missing:
            raise ValueError(f"unknown query id(s): {sorted(missing)}")
    if args.query_class:
        cases = [case for case in cases if case["class"] == args.query_class]
    if not cases:
        raise ValueError("no eval cases selected")

    created_at = datetime.now(timezone.utc)
    result_path, snapshot_path, latest_path, latest_snapshot_path = _run_paths(
        args.output, args.run_name, created_at,
    )
    snapshot_json = json.dumps(cases, indent=2)
    dataset_hash = hashlib.sha256(snapshot_json.encode()).hexdigest()
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(snapshot_json)

    settings = get_settings()
    es_client = get_es_client(settings)
    milvus_client = get_milvus_client(settings)
    gateway = GatewayClient(args.gateway_url or settings.gateway_url, trace_enabled=not args.no_langfuse)
    langfuse = get_client()
    try:
        results = []
        for index, case in enumerate(cases, start=1):
            print(f"[{index}/{len(cases)}] Running {case['id']} ({case['class']})...", flush=True)
            result = await evaluate_case(
                case, gateway, es_client, milvus_client, limit=args.limit,
                langfuse_enabled=not args.no_langfuse,
            )
            results.append(result)
            ranks = result["ranks"]
            print(
                f"[{index}/{len(cases)}] {case['id']} done: "
                f"ES={ranks['es'] or '>50'}, dense={ranks['raw_dense'] or '>50'}, "
                f"sparse={ranks['raw_sparse'] or '>50'}, RRF={ranks['rrf'] or '>50'}, "
                f"reranker={ranks['reranker'] or '>50'}",
                flush=True,
            )
        payload = {
            "run_name": args.run_name,
            "created_at": created_at.isoformat(),
            "git_revision": _git_revision(),
            "git_dirty": _git_dirty(),
            "dataset_sha256": dataset_hash,
            "dataset_snapshot": str(snapshot_path),
            "parameters": {
                "limit": args.limit,
                "query_ids": args.query,
                "query_class": args.query_class,
                "langfuse_enabled": not args.no_langfuse,
            },
            "results": results,
        }
        result_json = json.dumps(payload, indent=2)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(result_json)
        if latest_path is not None and latest_snapshot_path is not None:
            latest_path.write_text(result_json)
            latest_snapshot_path.write_text(snapshot_json)
        _print_summary(results)
        print(f"Full results: {result_path}")
        print(f"Dataset snapshot: {snapshot_path}")
        return 0
    finally:
        await es_client.close()
        milvus_client.close()
        if not args.no_langfuse:
            langfuse.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the case-law retrieval evaluation set")
    parser.add_argument("--dataset", type=Path, default=Path("evals/retrieval_cases.json"))
    parser.add_argument("--query", action="append", help="run one query ID; may be repeated")
    parser.add_argument("--class", dest="query_class", choices=["direct", "indirect", "adversarial"])
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--run-name", default="retrieval-eval")
    parser.add_argument("--gateway-url", help="override GATEWAY_URL (useful when running outside Docker)")
    parser.add_argument("--langfuse-base-url", help="override LANGFUSE_BASE_URL for host-side runs")
    parser.add_argument("--output", type=Path, help="exact result path; default creates a timestamped archive")
    parser.add_argument("--no-langfuse", action="store_true")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
