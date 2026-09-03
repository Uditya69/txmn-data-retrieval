from common.repotaxmannapi_token_dictionary import load_repotaxmannapi_token_dictionary


def test_loads_a_known_stopword_entry():
    entries = load_repotaxmannapi_token_dictionary()
    assert entries["A"] == {
        "element_type": "99", "tag_no": "0", "proximity": 2,
        "boost_factor": 0, "group_id": "0", "search_text": None,
    }


def test_loads_a_known_zone_entry_with_synonym_search_text():
    entries = load_repotaxmannapi_token_dictionary()
    assert entries["AAAR"] == {
        "element_type": "77", "tag_no": "0", "proximity": 2,
        "boost_factor": 0, "group_id": "0",
        "search_text": "Appellate Authority For Advance Ruling|AAAR",
    }


def test_loads_a_known_keyword_entry_with_real_group_id():
    entries = load_repotaxmannapi_token_dictionary()
    assert entries["ACCOUNTINGSTANDARD"] == {
        "element_type": "66", "tag_no": "2", "proximity": 0,
        "boost_factor": 0, "group_id": "111050000000011660",
        "search_text": "ACCOUNTING STANDARD",
    }


def test_dictionary_has_all_extracted_entries():
    entries = load_repotaxmannapi_token_dictionary()
    assert len(entries) > 2000
