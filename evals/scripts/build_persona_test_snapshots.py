"""Generates real, pipeline-derived persona snapshots for the rigorous keyword-
expansion eval (evals/scripts/keyword_expansion_rigorous_eval.py).

Writes synthetic query events through the REAL persona pipeline
(extract_query_understanding -> embed -> record_query_event, same path
persona_signal.py::record_persona_signal uses) for each topic in
_persona_topics.TOPICS, using distinct calendar-day timestamps and rich
interaction_signals so topics legitimately clear the active-state threshold
via the real state machine (persona.state_machine) rather than being
hand-set. Writes into the shared dev Mongo (taxmann_auth) under user_ids
prefixed "eval-persona-" - QA/dev environment, per project owner's explicit
go-ahead (2026-09-02) to use this DB directly and clean up after.

Usage:
    uv run python evals/scripts/build_persona_test_snapshots.py
    uv run python evals/scripts/build_persona_test_snapshots.py --cleanup   # deletes all eval-persona-* data
"""
import argparse
import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from _persona_topics import TOPICS

from common.config import get_settings
from persona.config import get_persona_settings
from persona.db import get_mongo_client, get_persona_events_collection, get_persona_topics_collection
from persona.prompt import render_persona_context
from persona.repository import get_current_snapshot, record_query_event
from retrieval_api.ai_mode.persona_signal import extract_query_understanding
from model_gateway.client import GatewayClient

RESULTS_DIR = Path(__file__).parent.parent / "results"
_BASE_DAY = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
# Rich signals so 4 distinct-day events clear ACTIVE_THRESHOLD (0.35) with
# 2-corroborating-session hysteresis at each upward transition (discovered ->
# emerging -> active) - see persona.state_machine/scoring for the math this
# is satisfying; plausible engagement (returned, repeated related query,
# saved), not fabricated to game the threshold.
_SIGNALS = {"submitted": True, "returned_later": True, "repeated_related_query": True, "saved": True}


async def build_topic(gateway, events, topics, topic: dict, settings) -> dict:
    user_id = topic["user_id"]
    topic_doc = None
    for day_index, query in enumerate(topic["queries"]):
        timestamp = _BASE_DAY + timedelta(days=day_index)
        understanding_raw = await extract_query_understanding(gateway, query)
        embedding = await gateway.embed(role="query_embed", text=query)
        topic_doc = await record_query_event(
            events, topics, user_id, query, understanding_raw, embedding,
            topic["categories"], _SIGNALS, timestamp, settings,
        )
    return topic_doc


async def _run(args) -> None:
    settings = get_settings()
    persona_settings = get_persona_settings()
    client = get_mongo_client(persona_settings)
    events = get_persona_events_collection(client, persona_settings)
    topics_coll = get_persona_topics_collection(client, persona_settings)
    gateway = GatewayClient(trace_enabled=False)

    # Topics are independent (distinct user_ids, no shared documents) - only the 4
    # events WITHIN a topic have a sequential dependency (each builds on the prior
    # score/state), so topics run concurrently while each topic's own build_topic
    # call stays sequential internally.
    final_docs = await asyncio.gather(*(
        build_topic(gateway, events, topics_coll, topic, persona_settings) for topic in TOPICS
    ))

    results = {}
    for topic, final_doc in zip(TOPICS, final_docs):
        snapshot = await get_current_snapshot(topics_coll, topic["user_id"])
        rendered = render_persona_context(snapshot, persona_settings)
        state = final_doc.get("state") if final_doc else None
        score = final_doc.get("score") if final_doc else None
        print(f"{topic['key']} ({topic['user_id']}): state={state} score={score} rendered={rendered!r}")
        results[topic["key"]] = {
            "user_id": topic["user_id"], "state": state, "score": score, "persona_context": rendered,
        }

    client.close()
    out_path = RESULTS_DIR / "persona_test_snapshots.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")

    non_active = [k for k, v in results.items() if not v["persona_context"]]
    if non_active:
        print(f"WARNING: these topics did not reach active/reactive above the confidence floor: {non_active}")


async def _cleanup() -> None:
    persona_settings = get_persona_settings()
    client = get_mongo_client(persona_settings)
    events = get_persona_events_collection(client, persona_settings)
    topics_coll = get_persona_topics_collection(client, persona_settings)
    user_ids = [t["user_id"] for t in TOPICS]
    ev_res = await events.delete_many({"user_id": {"$in": user_ids}})
    tp_res = await topics_coll.delete_many({"user_id": {"$in": user_ids}})
    client.close()
    print(f"Deleted {ev_res.deleted_count} events, {tp_res.deleted_count} topics for {len(user_ids)} eval-persona-* user_ids")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/cleanup real pipeline-derived persona snapshots for eval")
    parser.add_argument("--cleanup", action="store_true", help="delete all eval-persona-* events/topics and exit")
    args = parser.parse_args()
    if args.cleanup:
        asyncio.run(_cleanup())
    else:
        asyncio.run(_run(args))


if __name__ == "__main__":
    main()
