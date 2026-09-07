"""AI Mode reranker cap sweep.

Answers: how low can citations.py::_MAX_RERANK_CANDIDATES (currently 100) go
before recall drops? Sweeps a list of candidate caps against the same
pre-rerank RRF-merged candidate list per query, calling only the reranker at
each cap - ES, Milvus, and the SLM intent/rewrite call are never repeated,
they're read from the retrieval-eval stage cache instead (see
retrieval_api.retrieval_eval.stage_cache_path / --cache-dir). This makes each
sweep point cheap: the only per-cap cost is one DeepInfra rerank call per
query, on top of a fixed one-time cost to populate the cache.

Usage (from repo root, gateway running via docker compose):
    1. Populate the stage cache once (ES/Milvus/SLM calls happen here, and
       only here - this can be slow, run it once and reuse for every sweep):

        uv run retrieval-eval --cache-dir .rerank-cache --skip-synthesis \\
            --no-langfuse --gateway-url http://localhost:8001

    2. Sweep caps against that cache (fast - reranker calls only):

        uv run python evals/scripts/rerank_cap_sweep.py --cache-dir .rerank-cache \\
            --caps 100,50,25,20,10 --gateway-url http://localhost:8001

    Both commands must agree on --slm-model/--reranker-model/--sparse/--no-sparse
    (whatever isn't passed defaults the same way in both) - stage_cache_path()
    keys the cache file on these, so a mismatch is just a cache miss, not wrong
    data.

Writes one timestamped result file per run under .eval-results/ (so repeated
runs on a server don't clobber each other) plus a fixed
.eval-results/rerank_cap_sweep_latest.json pointing at the most recent run -
pass --output for an exact path instead. Prints a cap x recall/latency
summary table.
"""
import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from common.config import get_settings
from retrieval_api.ai_mode.rerank import rerank_top_chunks
from retrieval_api.gateway_client import GatewayClient
from retrieval_api.retrieval_eval import _git_dirty, _git_revision, doc_rank, load_cases, stage_cache_path

DEFAULT_CAPS = [100, 50, 25, 20, 10]


def _cap_candidates(merged: list[dict], cap: int) -> list[dict]:
    # Mirrors citations.py::rerank_and_prefetch's own cap logic exactly (sort
    # by rrf_score desc, slice to cap) so sweep results are comparable to
    # what AI Mode actually does in production at that cap value.
    return sorted(merged, key=lambda row: row["rrf_score"], reverse=True)[:cap]


async def _sweep_case(gateway, case: dict, merged: list[dict], caps: list[int], reranker_model: str | None) -> dict:
    gold = set(case["gold_doc_ids"])
    per_cap = {}
    for cap in caps:
        candidates = _cap_candidates(merged, cap)
        started = time.perf_counter()
        reranked = await rerank_top_chunks(gateway, case["query"], candidates, model=reranker_model)
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        rank = doc_rank(reranked, gold)
        per_cap[str(cap)] = {
            "rank": rank,
            "passed": rank is not None and rank <= case["pass_at"],
            "num_candidates": len(candidates),
            "latency_ms": latency_ms,
        }
    return per_cap


def _print_summary(caps: list[int], results: list[dict]) -> None:
    print(f"{'cap':>5}  {'recall@pass_at':>15}  {'avg_latency_ms':>15}")
    for cap in caps:
        key = str(cap)
        entries = [r["per_cap"][key] for r in results]
        passed = sum(e["passed"] for e in entries)
        avg_latency = round(sum(e["latency_ms"] for e in entries) / len(entries), 1)
        print(f"{cap:>5}  {passed:>7}/{len(entries):<7}  {avg_latency:>15}")


def _output_paths(output: Path | None, created_at: datetime) -> tuple[Path, Path | None]:
    if output is not None:
        return output, None
    timestamp = created_at.strftime("%Y%m%dT%H%M%S%fZ")
    return Path(".eval-results") / f"{timestamp}-rerank-cap-sweep.json", Path(".eval-results/rerank_cap_sweep_latest.json")


async def _run(args) -> int:
    cases = load_cases(args.dataset)
    settings = get_settings()
    sparse_enabled = settings.milvus_sparse_enabled if args.sparse_enabled is None else args.sparse_enabled
    gateway = GatewayClient(args.gateway_url or settings.gateway_url, trace_enabled=False)

    caps = sorted({int(c) for c in args.caps.split(",")}, reverse=True)
    created_at = datetime.now(timezone.utc)
    results = []
    for index, case in enumerate(cases, start=1):
        cache_path = stage_cache_path(
            args.cache_dir, case["id"], args.slm_model, args.reranker_model,
            rerank_enabled=True, sparse_enabled=sparse_enabled,
        )
        if not cache_path.exists():
            raise FileNotFoundError(
                f"no stage cache for {case['id']} at {cache_path} - run "
                "`retrieval-eval --cache-dir ...` first to populate it (see module docstring)",
            )
        cached = json.loads(cache_path.read_text())
        merged = cached["merged"]
        print(f"[{index}/{len(cases)}] {case['id']} ({len(merged)} cached candidates)...", flush=True)
        per_cap = await _sweep_case(gateway, case, merged, caps, args.reranker_model)
        results.append({"id": case["id"], "class": case["class"], "pass_at": case["pass_at"], "per_cap": per_cap})

    payload = {
        "created_at": created_at.isoformat(),
        "git_revision": _git_revision(),
        "git_dirty": _git_dirty(),
        "parameters": {
            "dataset": str(args.dataset),
            "cache_dir": str(args.cache_dir),
            "caps": caps,
            "slm_model": args.slm_model,
            "reranker_model": args.reranker_model,
            "sparse_enabled": sparse_enabled,
        },
        "caps": caps,
        "results": results,
    }
    output_path, latest_path = _output_paths(args.output, created_at)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload_json = json.dumps(payload, indent=2)
    output_path.write_text(payload_json)
    if latest_path is not None:
        latest_path.write_text(payload_json)
    print()
    _print_summary(caps, results)
    print(f"\nFull results: {output_path}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep AI Mode's reranker candidate cap against cached pre-rerank data")
    parser.add_argument("--dataset", type=Path, default=Path("evals/datasets/retrieval_cases.json"))
    parser.add_argument("--cache-dir", type=Path, required=True, help="stage cache populated by `retrieval-eval --cache-dir ...`")
    parser.add_argument("--caps", default=",".join(str(c) for c in DEFAULT_CAPS), help="comma-separated cap values, e.g. 100,50,25,20,10")
    parser.add_argument("--slm-model", help="must match the value used to populate --cache-dir")
    parser.add_argument("--reranker-model", help="model to rerank with; also must match --cache-dir's key if it was populated with an override")
    sparse_group = parser.add_mutually_exclusive_group()
    sparse_group.add_argument("--sparse", dest="sparse_enabled", action="store_true", default=None)
    sparse_group.add_argument("--no-sparse", dest="sparse_enabled", action="store_false")
    parser.add_argument("--gateway-url", help="override GATEWAY_URL (useful when running outside Docker)")
    parser.add_argument("--output", type=Path, help="exact result path; default creates a timestamped file under .eval-results/")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
