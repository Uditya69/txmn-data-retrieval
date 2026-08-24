import pytest

from semantic_cache.repository import lookup, write


@pytest.mark.asyncio
async def test_lookup_returns_none_when_empty(fake_semantic_cache_collection):
    result = await lookup(fake_semantic_cache_collection, "ai_mode", [1.0, 0.0], threshold=0.95)
    assert result is None


@pytest.mark.asyncio
async def test_write_then_lookup_exact_match_hits(fake_semantic_cache_collection):
    await write(
        fake_semantic_cache_collection, "ai_mode", "what is section 80C",
        [1.0, 0.0], {"ok": True, "answer": "cached answer", "citations": [], "intent": ["acts"]},
    )
    result = await lookup(fake_semantic_cache_collection, "ai_mode", [1.0, 0.0], threshold=0.95)
    assert result == {"ok": True, "answer": "cached answer", "citations": [], "intent": ["acts"]}


@pytest.mark.asyncio
async def test_lookup_below_threshold_misses(fake_semantic_cache_collection):
    await write(
        fake_semantic_cache_collection, "ai_mode", "what is section 80C",
        [1.0, 0.0], {"ok": True, "answer": "cached answer", "citations": [], "intent": []},
    )
    # Orthogonal vector -> cosine similarity 0.0, well below any reasonable threshold.
    result = await lookup(fake_semantic_cache_collection, "ai_mode", [0.0, 1.0], threshold=0.95)
    assert result is None


@pytest.mark.asyncio
async def test_lookup_is_scoped_by_mode(fake_semantic_cache_collection):
    await write(
        fake_semantic_cache_collection, "instant", "gst rate",
        [1.0, 0.0], {"es_error": None, "milvus_error": None, "es": [], "milvus": [], "milvus_sparse": []},
    )
    result = await lookup(fake_semantic_cache_collection, "ai_mode", [1.0, 0.0], threshold=0.95)
    assert result is None, "a doc cached under mode='instant' must not satisfy an 'ai_mode' lookup"


@pytest.mark.asyncio
async def test_lookup_returns_closest_of_multiple_candidates(fake_semantic_cache_collection):
    await write(fake_semantic_cache_collection, "ai_mode", "q1", [1.0, 0.0, 0.0], {"answer": "first"})
    await write(fake_semantic_cache_collection, "ai_mode", "q2", [0.99, 0.14, 0.0], {"answer": "second"})
    result = await lookup(fake_semantic_cache_collection, "ai_mode", [1.0, 0.0, 0.0], threshold=0.95)
    assert result == {"answer": "first"}
