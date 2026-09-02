from common.repotaxmannapi_tokenizer import classify_token


def test_classifies_a_known_stopword():
    entry = classify_token("a")
    assert entry is not None
    assert entry["element_type"] == "99"  # ElementType.StopWord


def test_classifies_a_known_court_entry_case_insensitively():
    entry = classify_token("AaR")
    assert entry is not None
    assert entry["element_type"] == "89"  # ElementType.Court


def test_returns_none_for_an_unrecognized_word():
    assert classify_token("zzznotarealword123") is None
