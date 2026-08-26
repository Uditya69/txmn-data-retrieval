import uuid
from datetime import datetime, timezone

from persona.clustering import find_best_topic, should_start_new_episode
from persona.evidence import evidence_weight
from persona.merge import KNOWN_CATEGORIES
from persona.scoring import interest_score
from persona.state_machine import next_state, reactivate_if_dormant
from persona.understanding import validate_understanding

# Legacy flat-document accessors (pre-timeline persona system, PR #5). Kept
# only so migrate_legacy_persona() can read a user's pre-existing document -
# no new code writes through these.


async def get_legacy_persona(personas, user_id: str) -> dict | None:
    return await personas.find_one({"user_id": user_id})


def _average_embedding(existing: list[float] | None, new: list[float]) -> list[float]:
    if not existing or len(existing) != len(new):
        return list(new)
    return [(a + b) / 2.0 for a, b in zip(existing, new)]


def _day_key(timestamp: datetime) -> str:
    return timestamp.date().isoformat()


async def _load_topics(topics, user_id: str) -> list[dict]:
    cursor = topics.find({"user_id": user_id})
    return [doc async for doc in cursor]


async def record_query_event(
    events, topics, user_id: str, query_text: str, understanding_raw: dict | None,
    embedding: list[float], categories: list[str], interaction_signals: dict | None,
    timestamp: datetime, settings,
) -> dict | None:
    """Appends one query event and derives the affected topic's updated state.

    Returns the updated topic document, or None if the Query Understanding
    Record failed validation (persona-query-understanding: an extraction
    failure produces no record at all, matching the pre-timeline
    implementation's failure-isolation contract).
    """
    validated = validate_understanding(understanding_raw)
    if validated is None:
        return None

    weight = evidence_weight(interaction_signals)
    entities_for_matching = validated["legal_entities"] + validated["concepts"]

    existing_topics = await _load_topics(topics, user_id)
    best_topic, match_score, match_signals = find_best_topic(
        existing_topics, embedding, entities_for_matching, categories, timestamp,
        settings.persona_cluster_similarity_threshold,
    )

    if best_topic is None:
        topic_id = str(uuid.uuid4())
        episode_id = str(uuid.uuid4())
        topic_doc = {
            "user_id": user_id,
            "topic_id": topic_id,
            "state": "discovered",
            "state_history": [{"state": "discovered", "entered_at": timestamp}],
            "episodes": [{"episode_id": episode_id, "started_at": timestamp, "ended_at": None}],
            "representative_embedding": embedding,
            "legal_entities": entities_for_matching[:20],
            "categories": list(categories),
            "last_event_at": timestamp,
            "sessions_since_transition": 1,
            "last_session_key": _day_key(timestamp),
            "score": 0.0,
        }
    else:
        topic_doc = dict(best_topic)
        topic_id = topic_doc["topic_id"]
        if should_start_new_episode(topic_doc.get("last_event_at"), timestamp, settings.persona_episode_gap_hours):
            topic_doc["episodes"] = list(topic_doc.get("episodes", [])) + [
                {"episode_id": str(uuid.uuid4()), "started_at": timestamp, "ended_at": None},
            ]
        episode_id = topic_doc["episodes"][-1]["episode_id"]
        topic_doc["representative_embedding"] = _average_embedding(topic_doc.get("representative_embedding"), embedding)
        topic_doc["legal_entities"] = list(set(topic_doc.get("legal_entities", []) + entities_for_matching))[:20]
        topic_doc["categories"] = list(set(topic_doc.get("categories", [])) | set(categories))

        day_key_now = _day_key(timestamp)
        if topic_doc.get("last_session_key") != day_key_now:
            topic_doc["sessions_since_transition"] = topic_doc.get("sessions_since_transition", 0) + 1
            topic_doc["last_session_key"] = day_key_now
        else:
            topic_doc.setdefault("sessions_since_transition", 1)

    event_doc = {
        "user_id": user_id,
        "query": query_text,
        "concepts": validated["concepts"],
        "legal_entities": validated["legal_entities"],
        "research_objective": validated["research_objective"],
        "specificity": validated["specificity"],
        "confidence": validated["confidence"],
        "categories": list(categories),
        "evidence_weight": weight,
        "topic_id": topic_id,
        "episode_id": episode_id,
        "timestamp": timestamp,
        # persona-topic-clustering's explainability requirement: keep the
        # winning similarity score/signal breakdown on the event itself so
        # explain_assignment() can answer "why" without recomputing anything.
        "clustering_score": match_score,
        "clustering_signals": match_signals,
        "is_new_topic": best_topic is None,
    }
    # Append-only: this insert never mutates or removes a prior event
    # (persona-timeline-storage's append-only requirement).
    await events.insert_one(event_doc)

    # KNOWN LIMITATION (noted, not fixed): this pulls the topic's ENTIRE event
    # history on every single write to recompute its score - unbounded compute
    # and Mongo read volume that grows with a topic's lifetime query count, not
    # just storage. Since interest_score() exponentially decays old events to
    # near-zero contribution anyway, a future fix should window this query
    # (last N events / last N decay-half-lives) instead of a full scan, and
    # likely pair it with a TTL index on persona_events for events old enough
    # to be scoring-irrelevant (mirrors auth's refresh_tokens TTL pattern in
    # persona/db.py). Flagged during review, deliberately deferred.
    topic_events = [doc async for doc in events.find({"user_id": user_id, "topic_id": topic_id})]
    score_events = [{"evidence_weight": e["evidence_weight"], "timestamp": e["timestamp"]} for e in topic_events]
    new_score = interest_score(score_events, timestamp, settings.persona_decay_lambda)

    current_state = topic_doc.get("state", "discovered")
    reactivated = reactivate_if_dormant(current_state, has_new_event=True)
    if reactivated != current_state:
        topic_doc["state_history"] = list(topic_doc.get("state_history", [])) + [
            {"state": reactivated, "entered_at": timestamp},
        ]
        topic_doc["sessions_since_transition"] = 1
        current_state = reactivated

    computed_next = next_state(
        current_state, new_score, topic_doc.get("sessions_since_transition", 0), settings.persona_min_sessions_per_transition,
    )
    if computed_next != current_state:
        topic_doc["state_history"] = list(topic_doc.get("state_history", [])) + [
            {"state": computed_next, "entered_at": timestamp},
        ]
        topic_doc["sessions_since_transition"] = 0
        current_state = computed_next

    topic_doc["state"] = current_state
    topic_doc["score"] = new_score
    topic_doc["last_event_at"] = timestamp

    await topics.replace_one({"user_id": user_id, "topic_id": topic_id}, topic_doc, upsert=True)
    return topic_doc


def explain_assignment(event: dict) -> dict:
    """Reports which topic/episode a query event was assigned to and which
    similarity signals contributed - persona-topic-clustering's
    explainability requirement. Takes the stored event document directly
    (produced by record_query_event); does not re-run clustering.
    """
    return {
        "topic_id": event.get("topic_id"),
        "episode_id": event.get("episode_id"),
        "is_new_topic": event.get("is_new_topic", False),
        "clustering_score": event.get("clustering_score", 0.0),
        "signals": event.get("clustering_signals", {}),
    }


async def get_current_snapshot(topics, user_id: str) -> list[dict]:
    """Derives the user's current persona from `persona_topics` - a snapshot,
    not a stored source of truth (persona-timeline-storage: recomputable,
    reproducible from the underlying history at any time).
    """
    all_topics = await _load_topics(topics, user_id)
    return sorted(all_topics, key=lambda t: (t.get("score", 0.0), t.get("last_event_at")), reverse=True)


_SEED_TOPIC_EVIDENCE_WEIGHT = 1.0


async def migrate_legacy_persona(events, topics, personas, user_id: str, timestamp: datetime, settings) -> bool:
    """One-time, lazy, on-first-touch conversion of a pre-timeline flat
    `personas` document into seed events - persona-timeline-storage's
    "existing flat persona documents are handled on transition" requirement.
    Never writes to or deletes from `personas` (design.md Migration Plan);
    read-only against the legacy collection.

    Returns True if a legacy document was found and converted, False if
    there was nothing to migrate (including: already migrated, since a
    migrated user has topics and this is only ever called when the new
    collections are empty for them).
    """
    legacy = await get_legacy_persona(personas, user_id)
    if not legacy:
        return False

    affinity = legacy.get("category_affinity") or {}
    seeded_any = False
    for category in KNOWN_CATEGORIES:
        weight = affinity.get(category, 0.0)
        if weight <= 0.0:
            continue
        topic_id = str(uuid.uuid4())
        episode_id = str(uuid.uuid4())
        event_doc = {
            "user_id": user_id,
            "query": "[migrated from legacy persona document]",
            "concepts": [],
            "legal_entities": [],
            "research_objective": [],
            "specificity": 0.0,
            "confidence": 0.0,
            "categories": [category],
            "evidence_weight": _SEED_TOPIC_EVIDENCE_WEIGHT * weight,
            "topic_id": topic_id,
            "episode_id": episode_id,
            "timestamp": timestamp,
        }
        await events.insert_one(event_doc)
        topic_doc = {
            "user_id": user_id,
            "topic_id": topic_id,
            "state": "discovered",
            "state_history": [{"state": "discovered", "entered_at": timestamp}],
            "episodes": [{"episode_id": episode_id, "started_at": timestamp, "ended_at": None}],
            "representative_embedding": [],
            "legal_entities": [],
            "categories": [category],
            "last_event_at": timestamp,
            "sessions_since_transition": 1,
            "last_session_key": _day_key(timestamp),
            "score": interest_score(
                [{"evidence_weight": event_doc["evidence_weight"], "timestamp": timestamp}], timestamp, settings.persona_decay_lambda,
            ),
        }
        await topics.replace_one({"user_id": user_id, "topic_id": topic_id}, topic_doc, upsert=True)
        seeded_any = True
    return seeded_any
