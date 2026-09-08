"""Probe for AI Mode's keyword-path SLM expansion (retrieval_api.ai_mode.keyword_expansion.
expand_keyword_terms), run before/after threading persona_context into it.

Motivating case: expand_keyword_terms's own system prompt refuses to guess on a bare
section/rule number with no Act/court/subject named ("never guess... a wrong guess
actively misdirects the lexical search"). A persona note naming the user's usual Act/
subject is exactly the kind of context that could responsibly clear that bar - this probe
measures whether threading persona_context through actually helps ES rank the correct doc
higher, using cases where the bare query alone is genuinely ambiguous across Acts.

Gold is regenerated live from ES each run (query + persona's Act/subject terms), same
"gold from live ES" convention as keyword_only_probe.py, not hand-fabricated doc_ids.

Usage:
    uv run python evals/scripts/keyword_expansion_probe.py
"""
import argparse
import asyncio
import inspect
import json
from pathlib import Path

from common.config import get_settings
from common.es_client import get_es_client, keyword_mode_search, raw_search
from retrieval_api.ai_mode.keyword_expansion import expand_keyword_terms
from model_gateway.client import GatewayClient
from retrieval_api.retrieval_eval import doc_rank

RESULTS_DIR = Path(__file__).parent.parent / "results"
LIMIT = 20
PASS_AT = 5

# Bare query alone is ambiguous across Acts; persona note supplies the Act/subject that
# resolves it. gold_hint is appended to the bare query only to regenerate gold from live
# ES - never shown to the model.
CASES = [
    {
        "id": "PK01", "query": "section 54", "gold_hint": "income-tax act capital gains house property",
        "persona_context": "User-focus note: this user's recent queries concern the Income-tax Act, "
                            "1961 - capital gains exemption on transfer of a residential house property.",
    },
    {
        "id": "PK02", "query": "section 43B", "gold_hint": "income-tax act disallowance unpaid statutory dues",
        "persona_context": "User-focus note: this user's recent queries concern the Income-tax Act, "
                            "1961 - disallowance of certain deductions on account of unpaid statutory dues.",
    },
    {
        "id": "PK03", "query": "rule 6", "gold_hint": "central goods and services tax rules input tax credit",
        "persona_context": "User-focus note: this user's recent queries concern the Central Goods and "
                            "Services Tax Rules - input tax credit conditions.",
    },
    {
        "id": "PK04", "query": "80HH", "gold_hint": "income-tax act newly established industrial undertakings deduction",
        "persona_context": "User-focus note: this user's recent queries concern the Income-tax Act, "
                            "1961 - deduction for newly established industrial undertakings in backward areas.",
    },
    {
        "id": "PK05", "query": "section 61", "gold_hint": "income-tax act revocable transfer of assets",
        "persona_context": "User-focus note: this user's recent queries concern the Income-tax Act, "
                            "1961 - clubbing of income from a revocable transfer of assets.",
    },
    {
        "id": "PK06", "query": "article 14", "gold_hint": "constitution of india right to equality",
        "persona_context": "User-focus note: this user's recent queries concern the Constitution of "
                            "India - the right to equality before law.",
    },
]


async def build_gold(es_client) -> list[dict]:
    cases = []
    for case in CASES:
        hits = await raw_search(es_client, f"{case['query']} {case['gold_hint']}", limit=1)
        if not hits:
            print(f"WARN: no live ES gold for {case['id']} {case['query']!r}, skipping")
            continue
        cases.append({**case, "gold_doc_id": hits[0]["doc_id"], "gold_heading": hits[0]["heading"]})
    return cases


async def run_case(gateway: GatewayClient, es_client, case: dict, use_persona: bool) -> dict:
    kwargs = {"persona_context": case["persona_context"]} if use_persona else {}
    added_keywords = await expand_keyword_terms(gateway, case["query"], **kwargs)
    keyword_query = case["query"] if not added_keywords else f"{case['query']} {' '.join(added_keywords)}"

    rows = await keyword_mode_search(es_client, keyword_query, doc_id_allowlist=None, limit=LIMIT)
    rank = doc_rank(rows, {case["gold_doc_id"]})
    return {
        "id": case["id"], "query": case["query"], "added_keywords": added_keywords,
        "keyword_query": keyword_query, "rank": rank, "hit": rank is not None and rank <= PASS_AT,
    }


async def _run(args) -> None:
    settings = get_settings()
    es_client = get_es_client(settings)
    gateway = GatewayClient(trace_enabled=False)

    supports_persona = "persona_context" in inspect.signature(expand_keyword_terms).parameters
    label = "after (persona-aware)" if supports_persona else "before (no persona support yet)"
    print(f"expand_keyword_terms persona_context support: {supports_persona} -> running {label}\n")

    try:
        cases = await build_gold(es_client)
        results = []
        for case in cases:
            result = await run_case(gateway, es_client, case, use_persona=supports_persona)
            results.append(result)
            print(
                f"{result['id']} {result['query']!r}: added={result['added_keywords']} "
                f"rank={result['rank'] if result['rank'] is not None else f'>{LIMIT}'} hit={result['hit']}"
            )
    finally:
        await es_client.close()

    hits = sum(r["hit"] for r in results)
    print(f"\nrecall@{PASS_AT}: {hits}/{len(results)}")

    out_path = RESULTS_DIR / f"keyword_expansion_probe_{'after' if supports_persona else 'before'}.json"
    out_path.write_text(json.dumps({"supports_persona": supports_persona, "results": results}, indent=2))
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe expand_keyword_terms persona-context impact")
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
