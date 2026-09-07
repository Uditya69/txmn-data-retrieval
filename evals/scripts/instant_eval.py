"""Instant mode fusion-strategy eval: off vs rrf vs reranker vs rrf+reranker.

Answers: for Instant mode's card list (run_instant()'s `reranked` output),
how does gold-doc_id recall/rank compare across its four candidate-fusion
strategies -

    - off:            instant/rerank.py::_fallback_fused (plain ES-ranked or
                      single-source list, today's default when neither toggle is on)
    - rrf:            rrf_merge_by_doc_id (ES + Milvus dense, rank-fused) - no cross-encoder
    - rerank:         cross-encoder over the plain ES+Milvus-dense union pool (RRF not
                      involved in candidate selection)
    - rrf_then_rerank: RRF fusion selects/orders the candidate pool first, *then* the
                      cross-encoder reranks whatever RRF picked (rerank_instant_results with
                      rrf=True, rerank=True - see _rrf_candidates in instant/rerank.py).
                      RRF's own ordering is discarded once the reranker re-sorts; only
                      *which* 20 candidates got selected differs from plain `rerank`.

Runs against real ES/Milvus (no stage cache - unlike AI Mode's
retrieval-eval, Instant's run_instant() bundles ES+Milvus+fuse into one call
with no natural seam to cache independently of the config being swept).

Usage (from repo root, gateway running via docker compose):

    uv run python evals/scripts/instant_eval.py --gateway-url http://localhost:8001

Writes one timestamped result file per run under .eval-results/ plus a fixed
.eval-results/instant_eval_latest.json pointing at the most recent run.
"""
import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from common.config import get_settings
from common.es_client import get_es_client
from common.milvus_client import get_milvus_client
from retrieval_api.gateway_client import GatewayClient
from retrieval_api.instant.search import run_instant
from retrieval_api.retrieval_eval import _git_dirty, _git_revision, doc_rank, load_cases

CONFIGS = {
    "off": {"rrf": False, "rerank": False},
    "rrf": {"rrf": True, "rerank": False},
    "rerank": {"rrf": False, "rerank": True},
    "rrf_then_rerank": {"rrf": True, "rerank": True},
}


async def _run_case(gateway, es_client, milvus_client, case: dict, boost_source: str, reranker_model: str | None) -> dict:
    gold = set(case["gold_doc_ids"])
    per_config = {}
    for name, toggles in CONFIGS.items():
        started = time.perf_counter()
        try:
            result = await run_instant(
                gateway, es_client, milvus_client, case["query"],
                rrf=toggles["rrf"], rerank=toggles["rerank"],
                boost_source=boost_source, reranker_model=reranker_model,
            )
            latency_ms = round((time.perf_counter() - started) * 1000, 1)
            reranked = result.get("reranked") or []
            rank = doc_rank(reranked, gold)
            per_config[name] = {
                "rank": rank,
                "passed": rank is not None and rank <= case["pass_at"],
                "num_results": len(reranked),
                "latency_ms": latency_ms,
                "es_error": result.get("es_error"),
                "milvus_error": result.get("milvus_error"),
                "reranked_error": result.get("reranked_error"),
            }
        except Exception as exc:  # noqa: BLE001 - one config's failure shouldn't kill the sweep
            per_config[name] = {
                "rank": None, "passed": False, "num_results": 0,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "error": f"{type(exc).__name__}: {exc}",
            }
    return {"id": case["id"], "class": case["class"], "pass_at": case["pass_at"], "query": case["query"], "per_config": per_config}


def _print_summary(results: list[dict]) -> None:
    print(f"\n{'config':>8}  {'recall@pass_at':>15}  {'avg_latency_ms':>15}  {'errors':>7}")
    for name in CONFIGS:
        entries = [r["per_config"][name] for r in results]
        passed = sum(e["passed"] for e in entries)
        errored = sum("error" in e for e in entries)
        latencies = [e["latency_ms"] for e in entries if "error" not in e]
        avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else float("nan")
        print(f"{name:>8}  {passed:>7}/{len(entries):<7}  {avg_latency:>15}  {errored:>7}")


def _output_paths(output: Path | None, created_at: datetime) -> tuple[Path, Path | None]:
    if output is not None:
        return output, None
    timestamp = created_at.strftime("%Y%m%dT%H%M%S%fZ")
    return Path(".eval-results") / f"{timestamp}-instant-eval.json", Path(".eval-results/instant_eval_latest.json")


async def _run(args) -> int:
    cases = load_cases(args.dataset)
    if args.query:
        wanted = set(args.query)
        cases = [case for case in cases if case["id"] in wanted]
    if args.query_class:
        cases = [case for case in cases if case["class"] == args.query_class]
    if not cases:
        raise ValueError("no eval cases selected")

    settings = get_settings()
    es_client = get_es_client(settings)
    milvus_client = get_milvus_client(settings)
    gateway = GatewayClient(args.gateway_url or settings.gateway_url, trace_enabled=False)

    created_at = datetime.now(timezone.utc)
    results = []
    try:
        for index, case in enumerate(cases, start=1):
            print(f"[{index}/{len(cases)}] {case['id']} ({case['class']})...", flush=True)
            result = await _run_case(gateway, es_client, milvus_client, case, args.boost_source, args.reranker_model)
            results.append(result)
            ranks = {name: entry.get("rank") for name, entry in result["per_config"].items()}
            print(f"[{index}/{len(cases)}] {case['id']} done: {ranks}", flush=True)

        payload = {
            "created_at": created_at.isoformat(),
            "git_revision": _git_revision(),
            "git_dirty": _git_dirty(),
            "dataset": str(args.dataset),
            "boost_source": args.boost_source,
            "reranker_model": args.reranker_model,
            "note": "rrf_then_rerank: RRF selects candidates, cross-encoder reranks them - see module docstring",
            "results": results,
        }
        result_path, latest_path = _output_paths(args.output, created_at)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        payload_json = json.dumps(payload, indent=2)
        result_path.write_text(payload_json)
        if latest_path is not None:
            latest_path.write_text(payload_json)
        _print_summary(results)
        print(f"\nFull results: {result_path}")
        return 0
    finally:
        await es_client.close()
        milvus_client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Instant mode off/rrf/reranker fusion-strategy eval")
    parser.add_argument("--dataset", type=Path, default=Path("evals/datasets/retrieval_cases.json"))
    parser.add_argument("--query", action="append", help="run one query ID; may be repeated")
    parser.add_argument("--class", dest="query_class", choices=["direct", "indirect", "adversarial"])
    parser.add_argument(
        "--boost-source", choices=["sum", "repotaxmannapi"], default="repotaxmannapi",
        help="matches run_instant()'s own default",
    )
    parser.add_argument("--reranker-model", help="override the DeepInfra model used for the reranker role (e.g. Qwen/Qwen3-Reranker-0.6B) - default is DEEPINFRA_RERANK_MODEL")
    parser.add_argument("--gateway-url", help="override GATEWAY_URL (useful when running outside Docker)")
    parser.add_argument("--output", type=Path, help="exact result path; default creates a timestamped archive")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
