from datetime import datetime, timedelta, timezone

from persona.clustering import cosine_similarity, find_best_topic, jaccard_overlap, should_start_new_episode

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
