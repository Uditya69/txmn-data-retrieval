"""Rigorous before/after eval for persona_context threading into AI Mode's keyword-
path SLM expansion (retrieval_api.ai_mode.keyword_expansion.expand_keyword_terms).

Fixes the three main weaknesses of the earlier quick probe
(keyword_expansion_probe.py):
  1. Non-circular: persona_context comes from evals/results/persona_test_snapshots.json,
     which is the REAL render_persona_context() output of REAL pipeline-derived
     topics (build_persona_test_snapshots.py) - not text authored to match the
     gold-generation hints below.
  2. Adversarial coverage: each bare query is also run against a MISMATCHED
     persona (a different topic's real persona) to verify the model does not
     get misdirected by irrelevant/conflicting context - the main safety risk
     called out for persona-driven SLM expansion.
  3. Variance: each case runs N times (temperature=0.6, non-deterministic).

Gold doc_ids are regenerated live from ES per run (query + independently-
authored hint terms - never copied from the persona text), same "gold from
live ES" convention as keyword_only_probe.py.

Usage:
    uv run python evals/scripts/keyword_expansion_rigorous_eval.py
"""
import argparse
import asyncio
import json
from pathlib import Path

from common.config import get_settings
from common.es_client import get_es_client, keyword_mode_search, raw_search
from retrieval_api.ai_mode.keyword_expansion import expand_keyword_terms

_MAX_RETRIES = 4
_RETRY_BASE_DELAY = 3.0


async def _with_retries(coro_fn, *args, **kwargs):
    """Retries a flaky network call (ES/gateway) with exponential backoff - the
    live ES cluster this eval hits over an unstable connection has dropped mid-run
    twice already, each time discarding 20-40+ minutes of already-completed SLM
    calls. A transient timeout/connection error should not cost the whole run."""
    last_exc = None
    for attempt in range(_MAX_RETRIES):
        try:
            return await coro_fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - retry any transient transport failure
            last_exc = exc
            if attempt < _MAX_RETRIES - 1:
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                print(f"  (retry {attempt + 1}/{_MAX_RETRIES - 1} after {type(exc).__name__}: {exc}; waiting {delay:.0f}s)")
                await asyncio.sleep(delay)
    raise last_exc
from model_gateway.client import GatewayClient
from retrieval_api.retrieval_eval import doc_rank

RESULTS_DIR = Path(__file__).parent.parent / "results"
LIMIT = 20
PASS_AT = 5
RUNS_PER_CASE = 3

# bare query + independently-authored gold hint (never copied from persona text)
# + the mismatched topic key used for that query's adversarial pairing, and the
# "wrong-signal" terms that would indicate the model leaked the mismatched
# persona's Act/subject into its added keywords.
QUERY_SPECS = {
    "sec54": {
        "query": "section 54", "gold_hint": "income-tax act capital gains house property",
    },
    "sec43b": {
        "query": "section 43B", "gold_hint": "income-tax act disallowance unpaid statutory dues",
    },
    "gst_rule6": {
        "query": "rule 6", "gold_hint": "central goods and services tax rules input tax credit",
    },
    "sec80hh": {
        "query": "80HH", "gold_hint": "income-tax act newly established industrial undertakings deduction",
    },
    "sec61": {
        "query": "section 61", "gold_hint": "income-tax act revocable transfer of assets",
    },
    "art14": {
        "query": "article 14", "gold_hint": "constitution of india right to equality",
    },
}

# distinctive terms per topic - present in the topic's OWN correct expansion,
# and a leakage signal if they show up in a DIFFERENT topic's added_keywords.
SIGNAL_TERMS = {
    "sec54": ["section 54", "house property", "capital gains"],
    "sec43b": ["43b", "statutory dues", "bonus"],
    "gst_rule6": ["cgst", "rule 6", "input tax credit", "itc", "gst"],
    "sec80hh": ["80hh", "industrial undertaking", "backward area"],
    "sec61": ["section 61", "revocable", "clubbing"],
    "art14": ["article 14", "constitution", "equality"],
}

# Adversarial pairing: each topic's bare query is also run against a DIFFERENT,
# unrelated topic's real persona (round-robin shift) - the model should ignore it.
_KEYS = list(QUERY_SPECS.keys())
MISMATCH = {key: _KEYS[(i + 1) % len(_KEYS)] for i, key in enumerate(_KEYS)}


def _leaked(added_keywords: list[str], terms: list[str]) -> bool:
    joined = " ".join(added_keywords).lower()
    return any(term in joined for term in terms)


async def build_gold(es_client) -> dict:
    gold = {}
    for key, spec in QUERY_SPECS.items():
        hits = await _with_retries(raw_search, es_client, f"{spec['query']} {spec['gold_hint']}", limit=1)
        if not hits:
            print(f"WARN: no live ES gold for {key} {spec['query']!r}")
            continue
        gold[key] = hits[0]["doc_id"]
    return gold


async def run_once(gateway, es_client, query: str, persona_context: str) -> dict:
    kwargs = {"persona_context": persona_context} if persona_context else {}
    added_keywords = await _with_retries(expand_keyword_terms, gateway, query, **kwargs)
    keyword_query = query if not added_keywords else f"{query} {' '.join(added_keywords)}"
    rows = await _with_retries(keyword_mode_search, es_client, keyword_query, doc_id_allowlist=None, limit=LIMIT)
    return {"added_keywords": added_keywords, "keyword_query": keyword_query, "rows": rows}


async def run_case(gateway, es_client, key: str, persona_context: str, gold_doc_id: str, n: int) -> dict:
    query = QUERY_SPECS[key]["query"]
    runs = []
    for _ in range(n):
        result = await run_once(gateway, es_client, query, persona_context)
        rank = doc_rank(result["rows"], {gold_doc_id})
        runs.append({
            "added_keywords": result["added_keywords"], "keyword_query": result["keyword_query"],
            "rank": rank, "hit": rank is not None and rank <= PASS_AT,
        })
    return {"key": key, "query": query, "runs": runs}


async def _run(args) -> None:
    settings = get_settings()
    es_client = get_es_client(settings)
    gateway = GatewayClient(trace_enabled=False)

    personas = json.loads((RESULTS_DIR / "persona_test_snapshots.json").read_text())
    personas = {k: v["persona_context"] for k, v in personas.items() if v["persona_context"]}
    missing = [k for k in QUERY_SPECS if k not in personas]
    if missing:
        raise SystemExit(f"Missing active persona_context for: {missing} - rerun build_persona_test_snapshots.py")

    # Checkpointing: the live ES connection this eval runs over has dropped mid-run
    # twice already, each time discarding 20-40+ minutes of already-completed SLM
    # calls. Write the partial report to disk after every case and, on the next
    # invocation, skip any (section, key) already present there instead of
    # redoing it - a crash costs at most one in-flight case, not the whole run.
    checkpoint_path = RESULTS_DIR / "keyword_expansion_rigorous_results.partial.json"
    if checkpoint_path.exists():
        report = json.loads(checkpoint_path.read_text())
        print(f"Resuming from checkpoint {checkpoint_path} (sections so far: "
              f"{[k for k, v in report.items() if v]})")
    else:
        report = {"before": [], "after_own_persona": [], "after_mismatched_persona": []}

    def done_keys(section: str) -> set:
        return {c["key"] for c in report[section]}

    def checkpoint() -> None:
        checkpoint_path.write_text(json.dumps(report, indent=2))

    try:
        gold = await build_gold(es_client)

        print("=== BEFORE (no persona) ===")
        for key in QUERY_SPECS:
            if key in done_keys("before"):
                continue
            case = await run_case(gateway, es_client, key, "", gold[key], RUNS_PER_CASE)
            hits = sum(r["hit"] for r in case["runs"])
            print(f"{key} {case['query']!r}: hits={hits}/{RUNS_PER_CASE} added={[r['added_keywords'] for r in case['runs']]}")
            report["before"].append(case)
            checkpoint()

        print("\n=== AFTER, own (matching) persona ===")
        for key in QUERY_SPECS:
            if key in done_keys("after_own_persona"):
                continue
            case = await run_case(gateway, es_client, key, personas[key], gold[key], RUNS_PER_CASE)
            hits = sum(r["hit"] for r in case["runs"])
            print(f"{key} {case['query']!r}: hits={hits}/{RUNS_PER_CASE} added={[r['added_keywords'] for r in case['runs']]}")
            report["after_own_persona"].append(case)
            checkpoint()

        print("\n=== AFTER, mismatched (adversarial) persona ===")
        for key in QUERY_SPECS:
            if key in done_keys("after_mismatched_persona"):
                continue
            wrong_key = MISMATCH[key]
            case = await run_case(gateway, es_client, key, personas[wrong_key], gold[key], RUNS_PER_CASE)
            case["mismatched_persona_topic"] = wrong_key
            leaks = sum(_leaked(r["added_keywords"], SIGNAL_TERMS[wrong_key]) for r in case["runs"])
            hits = sum(r["hit"] for r in case["runs"])
            print(
                f"{key} {case['query']!r} vs {wrong_key} persona: hits={hits}/{RUNS_PER_CASE} "
                f"leaked_wrong_topic={leaks}/{RUNS_PER_CASE} added={[r['added_keywords'] for r in case['runs']]}"
            )
            report["after_mismatched_persona"].append(case)
            checkpoint()
    finally:
        await es_client.close()

    def summarize(cases):
        total_runs = sum(len(c["runs"]) for c in cases)
        total_hits = sum(sum(r["hit"] for r in c["runs"]) for c in cases)
        return total_hits, total_runs

    before_hits, before_n = summarize(report["before"])
    after_hits, after_n = summarize(report["after_own_persona"])
    mismatch_hits, mismatch_n = summarize(report["after_mismatched_persona"])
    total_leaks = sum(
        _leaked(r["added_keywords"], SIGNAL_TERMS[c["mismatched_persona_topic"]])
        for c in report["after_mismatched_persona"] for r in c["runs"]
    )

    print("\n=== SUMMARY ===")
    print(f"before (no persona):            {before_hits}/{before_n} run-hits @pass_at={PASS_AT}")
    print(f"after (own/matching persona):   {after_hits}/{after_n} run-hits @pass_at={PASS_AT}")
    print(f"after (mismatched persona):     {mismatch_hits}/{mismatch_n} run-hits @pass_at={PASS_AT} "
          f"| wrong-topic leakage: {total_leaks}/{mismatch_n} runs")

    out_path = RESULTS_DIR / "keyword_expansion_rigorous_results.json"
    out_path.write_text(json.dumps(report, indent=2))
    checkpoint_path.unlink(missing_ok=True)
    print(f"\nWrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rigorous persona-aware keyword-expansion eval")
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
