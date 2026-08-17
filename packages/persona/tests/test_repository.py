import pytest

from persona.repository import get_persona, record_signal


@pytest.mark.asyncio
async def test_get_persona_returns_none_when_absent(fake_personas_collection):
    personas = fake_personas_collection
    assert await get_persona(personas, "user-1") is None


@pytest.mark.asyncio
async def test_record_signal_creates_persona_on_first_call(fake_personas_collection):
    personas = fake_personas_collection
    result = await record_signal(personas, "user-1", categories=["caselaws"], expertise_patch={"expertise_level": "student"})
    assert result["user_id"] == "user-1"
    assert result["category_affinity"]["caselaws"] == 1.0
    assert result["expertise_level"] == "student"
    assert result["query_count"] == 1


@pytest.mark.asyncio
async def test_record_signal_merges_into_existing_persona(fake_personas_collection):
    personas = fake_personas_collection
    await record_signal(personas, "user-1", categories=["caselaws"], expertise_patch=None)
    result = await record_signal(personas, "user-1", categories=["acts"], expertise_patch={"query_style": "precise-citation"})
    assert result["category_affinity"]["caselaws"] == 0.5
    assert result["category_affinity"]["acts"] == 0.5
    assert result["query_style"] == "precise-citation"
    assert result["query_count"] == 2


@pytest.mark.asyncio
async def test_get_persona_returns_stored_document(fake_personas_collection):
    personas = fake_personas_collection
    await record_signal(personas, "user-1", categories=["commentary"], expertise_patch=None)
    stored = await get_persona(personas, "user-1")
    assert stored["category_affinity"]["commentary"] == 1.0
