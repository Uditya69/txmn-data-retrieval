from common.instant_classifier.labels import (
    HYBRID, INTENT, KEYWORD,
    ClassifierResult, boost_profile_key, resolve_routing, routing_plan,
)


def test_boost_profile_key_maps_each_label_to_itself():
    assert boost_profile_key(KEYWORD) == "KEYWORD"
    assert boost_profile_key(HYBRID) == "HYBRID"
    assert boost_profile_key(INTENT) == "INTENT"


def test_resolve_routing_keeps_label_when_confident():
    result = ClassifierResult(label=KEYWORD, confidence=0.95)
    assert resolve_routing(result, threshold=0.5) == KEYWORD


def test_resolve_routing_defaults_to_hybrid_when_below_threshold():
    result = ClassifierResult(label=KEYWORD, confidence=0.4)
    assert resolve_routing(result, threshold=0.5) == HYBRID


def test_routing_plan_keyword_skips_milvus_no_fusion():
    assert routing_plan(KEYWORD) == {"es": True, "milvus": False, "fuse": False}


def test_routing_plan_intent_skips_es_no_fusion():
    assert routing_plan(INTENT) == {"es": False, "milvus": True, "fuse": False}


def test_routing_plan_hybrid_queries_both_and_fuses():
    assert routing_plan(HYBRID) == {"es": True, "milvus": True, "fuse": True}
