from datetime import datetime, timedelta, timezone

import pytest

from persona.clustering import cosine_similarity, entity_similarity, find_best_topic, jaccard_overlap, should_start_new_episode

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_cosine_similarity_identical_vectors():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_cosine_similarity_orthogonal_vectors():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_similarity_empty_or_mismatched_is_zero():
    assert cosine_similarity([], [1.0]) == 0.0
    assert cosine_similarity([1.0, 0.0], [1.0]) == 0.0


def test_jaccard_overlap_full_and_none():
    assert jaccard_overlap(["IBC"], ["ibc"]) == 1.0
    assert jaccard_overlap(["IBC"], ["GST"]) == 0.0
    assert jaccard_overlap([], []) == 0.0


def test_find_best_topic_matches_related_query():
    existing = [{
        "topic_id": "t1", "representative_embedding": [1.0, 0.0, 0.0],
        "legal_entities": ["IBC"], "categories": ["caselaws"], "last_event_at": T0,
    }]
    topic, score, signals = find_best_topic(
        existing, [0.99, 0.1, 0.0], ["IBC"], ["caselaws"], T0 + timedelta(minutes=5), threshold=0.6,
    )
    assert topic["topic_id"] == "t1"
    assert score >= 0.6
    assert signals["embedding_similarity"] > 0.9


def test_find_best_topic_returns_none_below_threshold():
    existing = [{
        "topic_id": "t1", "representative_embedding": [1.0, 0.0, 0.0],
        "legal_entities": ["IBC"], "categories": ["caselaws"], "last_event_at": T0,
    }]
    topic, score, _signals = find_best_topic(
        existing, [0.0, 1.0, 0.0], ["GST"], ["acts"], T0 + timedelta(days=90), threshold=0.6,
    )
    assert topic is None
    assert score < 0.6


def test_find_best_topic_empty_candidates_returns_none():
    topic, score, _signals = find_best_topic([], [1.0], ["IBC"], [], T0, threshold=0.6)
    assert topic is None
    assert score == 0.0


def test_should_start_new_episode_true_when_no_prior_event():
    assert should_start_new_episode(None, T0, episode_gap_hours=168.0) is True


def test_should_start_new_episode_false_within_gap():
    assert should_start_new_episode(T0, T0 + timedelta(hours=5), episode_gap_hours=168.0) is False


def test_should_start_new_episode_true_beyond_gap():
    assert should_start_new_episode(T0, T0 + timedelta(days=30), episode_gap_hours=168.0) is True


def test_entity_similarity_tolerates_acronym_and_parenthetical_variance():
    # Real bug found during end-to-end testing: the SLM phrases the same
    # legal instrument differently across separate extraction calls.
    a = ["Insolvency and Bankruptcy Code (IBC)", "personal liability of directors", "corporate debt"]
    b = ["Financial Creditor", "Section 7 application", "Insolvency and Bankruptcy Code"]
    assert entity_similarity(a, b) > 0.0


def test_entity_similarity_matches_bare_acronym_against_full_name_with_parenthetical():
    a = ["Insolvency and Bankruptcy Code (IBC)"]
    b = ["IBC"]
    assert entity_similarity(a, b) > 0.0


def test_entity_similarity_exact_match_scores_higher_than_partial():
    identical = entity_similarity(["Insolvency and Bankruptcy Code (IBC)"], ["Insolvency and Bankruptcy Code (IBC)"])
    partial = entity_similarity(["Insolvency and Bankruptcy Code (IBC)"], ["Goods and Services Tax (GST)"])
    assert identical == 1.0
    assert partial < identical


def test_entity_similarity_unrelated_entities_scores_zero():
    assert entity_similarity(["Insolvency and Bankruptcy Code"], ["Goods and Services Tax"]) == 0.0


def test_find_best_topic_matches_related_query_via_normalized_entities_not_just_embedding():
    # Regression for the real end-to-end finding (measured against live
    # model-gateway calls, not simulated): embedding cosine similarity
    # between two genuinely related but differently-phrased IBC queries
    # measured only ~0.32 - Voyage's query_embed is tuned for query->document
    # retrieval matching, not query->query similarity. Combined score for
    # this exact real pair, with entity normalization applied, is ~0.446 -
    # this test locks in that it clears the (also-recalibrated) 0.4 default
    # threshold.
    existing = [{
        "topic_id": "t1",
        "representative_embedding": [1.0, 0.0, 0.0],
        "legal_entities": ["Insolvency and Bankruptcy Code (IBC)", "personal liability of directors", "corporate debt"],
        "categories": ["caselaws"],
        "last_event_at": T0,
    }]
    topic, score, signals = find_best_topic(
        existing, [0.32, 0.947, 0.0],  # cosine ~0.32 against [1,0,0], as actually measured
        ["Financial Creditor", "Section 7 application", "Insolvency and Bankruptcy Code"],
        ["caselaws"], T0 + timedelta(minutes=5), threshold=0.4,
    )
    assert topic is not None
    assert topic["topic_id"] == "t1"
    assert signals["entity_overlap"] > 0.0
    assert score == pytest.approx(0.446, abs=0.01)
