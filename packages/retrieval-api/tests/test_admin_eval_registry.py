import logging

from retrieval_api.admin_eval.registry import SUITES


def test_all_suites_registered():
    assert set(SUITES.keys()) == {"slm_intent", "collection_routing", "retrieval"}


def test_each_suite_has_a_display_name_and_callable_run():
    for suite_id, suite in SUITES.items():
        assert isinstance(suite["name"], str) and suite["name"]
        assert callable(suite["run"])


def test_langfuse_logger_silenced_on_import():
    assert logging.getLogger("langfuse").level == logging.CRITICAL
