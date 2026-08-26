from datetime import datetime, timedelta, timezone

import pytest

from persona.repository import explain_assignment, get_current_snapshot, migrate_legacy_persona, record_query_event

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _understanding(**overrides):
    base = {
        "concepts": ["director liability"],
        "legal_entities": ["IBC"],
        "research_objective": ["determine liability"],
        "specificity": 0.8,
        "confidence": 0.9,
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_record_query_event_returns_none_on_failed_extraction(fake_events_collection, fake_topics_collection, persona_settings):
    result = await record_query_event(
        fake_events_collection, fake_topics_collection, "user-1", "query text", None,
        [0.1, 0.2], ["caselaws"], {"submitted": True}, T0, persona_settings,
    )
    assert result is None
    assert fake_events_collection.documents == []


@pytest.mark.asyncio
async def test_record_query_event_creates_new_discovered_topic(fake_events_collection, fake_topics_collection, persona_settings):
    result = await record_query_event(
        fake_events_collection, fake_topics_collection, "user-1", "Section 7 IBC", _understanding(),
        [0.1, 0.2, 0.3], ["caselaws"], {"submitted": True}, T0, persona_settings,
    )
    assert result["state"] == "discovered"
    assert result["user_id"] == "user-1"
    assert len(fake_events_collection.documents) == 1


@pytest.mark.asyncio
async def test_record_query_event_appends_without_mutating_prior_events(fake_events_collection, fake_topics_collection, persona_settings):
    await record_query_event(
        fake_events_collection, fake_topics_collection, "user-1", "Section 7 IBC", _understanding(),
        [0.1, 0.2, 0.3], ["caselaws"], {"submitted": True}, T0, persona_settings,
    )
    first_event = dict(fake_events_collection.documents[0])
    await record_query_event(
        fake_events_collection, fake_topics_collection, "user-1", "financial creditor under section 7",
        _understanding(concepts=["financial creditor"]),
        [0.11, 0.21, 0.31], ["caselaws"], {"submitted": True}, T0 + timedelta(minutes=5), persona_settings,
    )
    assert fake_events_collection.documents[0] == first_event
    assert len(fake_events_collection.documents) == 2


@pytest.mark.asyncio
async def test_record_query_event_clusters_related_queries_into_same_topic(fake_events_collection, fake_topics_collection, persona_settings):
    first = await record_query_event(
        fake_events_collection, fake_topics_collection, "user-1", "Section 7 IBC", _understanding(),
        [0.1, 0.2, 0.3], ["caselaws"], {"submitted": True}, T0, persona_settings,
    )
    second = await record_query_event(
        fake_events_collection, fake_topics_collection, "user-1", "financial creditor under section 7",
        _understanding(concepts=["financial creditor"]),
        [0.1, 0.2, 0.31], ["caselaws"], {"submitted": True}, T0 + timedelta(minutes=5), persona_settings,
    )
    assert second["topic_id"] == first["topic_id"]


@pytest.mark.asyncio
async def test_record_query_event_starts_new_topic_for_unrelated_query(fake_events_collection, fake_topics_collection, persona_settings):
    first = await record_query_event(
        fake_events_collection, fake_topics_collection, "user-1", "Section 7 IBC", _understanding(),
        [1.0, 0.0, 0.0], ["caselaws"], {"submitted": True}, T0, persona_settings,
    )
    second = await record_query_event(
        fake_events_collection, fake_topics_collection, "user-1", "GST registration cancellation",
        _understanding(concepts=["registration cancellation"], legal_entities=["GST"]),
        [0.0, 1.0, 0.0], ["acts"], {"submitted": True}, T0 + timedelta(days=90), persona_settings,
    )
    assert second["topic_id"] != first["topic_id"]


@pytest.mark.asyncio
async def test_record_query_event_promotes_topic_after_sustained_sessions(fake_events_collection, fake_topics_collection, persona_settings):
    topic = None
    for day in range(10):
        topic = await record_query_event(
            fake_events_collection, fake_topics_collection, "user-1", "Section 7 IBC", _understanding(),
            [1.0, 0.0, 0.0], ["caselaws"], {"submitted": True, "saved": True}, T0 + timedelta(days=day), persona_settings,
        )
    assert topic["state"] in ("emerging", "active")


@pytest.mark.asyncio
async def test_record_query_event_single_session_spike_does_not_promote_to_active(fake_events_collection, fake_topics_collection, persona_settings):
    topic = await record_query_event(
        fake_events_collection, fake_topics_collection, "user-1", "Section 7 IBC", _understanding(),
        [1.0, 0.0, 0.0], ["caselaws"], {"submitted": True, "saved": True, "returned_later": True}, T0, persona_settings,
    )
    assert topic["state"] != "active"


@pytest.mark.asyncio
async def test_explain_assignment_reports_topic_and_signals(fake_events_collection, fake_topics_collection, persona_settings):
    await record_query_event(
        fake_events_collection, fake_topics_collection, "user-1", "Section 7 IBC", _understanding(),
        [1.0, 0.0, 0.0], ["caselaws"], {"submitted": True}, T0, persona_settings,
    )
    event = fake_events_collection.documents[0]
    explanation = explain_assignment(event)
    assert explanation["topic_id"] == event["topic_id"]
    assert explanation["is_new_topic"] is True
    assert "signals" in explanation


@pytest.mark.asyncio
async def test_record_query_event_reactivates_dormant_topic_preserving_history(
    fake_events_collection, fake_topics_collection, persona_settings,
):
    discovered_at = T0
    dormant_doc = {
        "user_id": "user-1",
        "topic_id": "gst-topic",
        "state": "dormant",
        "state_history": [
            {"state": "discovered", "entered_at": discovered_at},
            {"state": "dormant", "entered_at": T0 + timedelta(days=180)},
        ],
        "episodes": [{"episode_id": "ep-1", "started_at": discovered_at, "ended_at": None}],
        "representative_embedding": [0.0, 1.0, 0.0],
        "legal_entities": ["GST"],
        "categories": ["acts"],
        "last_event_at": T0 + timedelta(days=180),
        "sessions_since_transition": 0,
        "last_session_key": "2026-06-30",
        "score": 0.02,
    }
    await fake_topics_collection.replace_one({"user_id": "user-1", "topic_id": "gst-topic"}, dormant_doc, upsert=True)

    result = await record_query_event(
        fake_events_collection, fake_topics_collection, "user-1", "GST registration cancellation",
        _understanding(concepts=["registration"], legal_entities=["GST"]),
        [0.0, 1.0, 0.0], ["acts"], {"submitted": True}, T0 + timedelta(days=360), persona_settings,
    )

    assert result["topic_id"] == "gst-topic"
    assert result["state"] == "reactive"
    assert result["episodes"][0]["episode_id"] == "ep-1"
    assert result["state_history"][0]["state"] == "discovered"
    assert result["state_history"][0]["entered_at"] == discovered_at


@pytest.mark.asyncio
async def test_record_query_event_separates_non_adjacent_activity_into_two_episodes(
    fake_events_collection, fake_topics_collection, persona_settings,
):
    first = await record_query_event(
        fake_events_collection, fake_topics_collection, "user-1", "GST input tax credit",
        _understanding(concepts=["input tax credit"], legal_entities=["GST"]),
        [0.0, 1.0, 0.0], ["acts"], {"submitted": True}, T0, persona_settings,
    )
    assert len(first["episodes"]) == 1

    second = await record_query_event(
        fake_events_collection, fake_topics_collection, "user-1", "GST registration cancellation",
        _understanding(concepts=["registration cancellation"], legal_entities=["GST"]),
        [0.0, 1.0, 0.0], ["acts"], {"submitted": True}, T0 + timedelta(days=60), persona_settings,
    )

    assert second["topic_id"] == first["topic_id"]
    assert len(second["episodes"]) == 2
    assert second["episodes"][0]["episode_id"] != second["episodes"][1]["episode_id"]


@pytest.mark.asyncio
async def test_get_current_snapshot_returns_empty_for_new_user(fake_topics_collection):
    assert await get_current_snapshot(fake_topics_collection, "user-1") == []


@pytest.mark.asyncio
async def test_get_current_snapshot_reflects_recorded_topics(fake_events_collection, fake_topics_collection, persona_settings):
    await record_query_event(
        fake_events_collection, fake_topics_collection, "user-1", "Section 7 IBC", _understanding(),
        [1.0, 0.0, 0.0], ["caselaws"], {"submitted": True}, T0, persona_settings,
    )
    snapshot = await get_current_snapshot(fake_topics_collection, "user-1")
    assert len(snapshot) == 1
    assert snapshot[0]["user_id"] == "user-1"


@pytest.mark.asyncio
async def test_get_current_snapshot_is_reproducible_from_the_same_history(fake_events_collection, fake_topics_collection, persona_settings):
    await record_query_event(
        fake_events_collection, fake_topics_collection, "user-1", "Section 7 IBC", _understanding(),
        [1.0, 0.0, 0.0], ["caselaws"], {"submitted": True}, T0, persona_settings,
    )
    first = await get_current_snapshot(fake_topics_collection, "user-1")
    second = await get_current_snapshot(fake_topics_collection, "user-1")
    assert first == second


@pytest.mark.asyncio
async def test_migrate_legacy_persona_seeds_events_without_touching_legacy_doc(
    fake_events_collection, fake_topics_collection, fake_personas_collection, persona_settings,
):
    await fake_personas_collection.replace_one(
        {"user_id": "user-1"},
        {"user_id": "user-1", "category_affinity": {"caselaws": 0.8, "acts": 0.2}},
        upsert=True,
    )
    legacy_before = dict(await fake_personas_collection.find_one({"user_id": "user-1"}))

    migrated = await migrate_legacy_persona(
        fake_events_collection, fake_topics_collection, fake_personas_collection, "user-1", T0, persona_settings,
    )

    assert migrated is True
    assert len(fake_events_collection.documents) == 2
    assert await fake_personas_collection.find_one({"user_id": "user-1"}) == legacy_before


@pytest.mark.asyncio
async def test_migrate_legacy_persona_returns_false_when_nothing_to_migrate(
    fake_events_collection, fake_topics_collection, fake_personas_collection, persona_settings,
):
    migrated = await migrate_legacy_persona(
        fake_events_collection, fake_topics_collection, fake_personas_collection, "user-1", T0, persona_settings,
    )
    assert migrated is False
