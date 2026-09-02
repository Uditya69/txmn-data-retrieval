from common.repotaxmannapi_query_builder import build_should_clauses
from common.repotaxmannapi_tokenizer import RepotaxmannapiToken


def test_builds_phrase_boost_should_clauses_for_a_plain_text_token():
    token = RepotaxmannapiToken(
        query_text="Dimension Data India", org_text="Dimension Data India",
        type="TX", or_in=False, proximity=5, query_date=None,
    )
    clauses = build_should_clauses([token], is_global=True, is_excus=False)

    boosts_by_field = {
        list(c["match_phrase"].keys())[0]: list(c["match_phrase"].values())[0]["boost"]
        for c in clauses if "match_phrase" in c
    }
    assert boosts_by_field == {
        "heading": 155000, "subheading": 80000,
        "searchboosttext": 70000, "headnotestext": 65000, "fullcontent": 1,
    }


def test_adds_sub_exclusion_clause_for_a_section_prefixed_token():
    token = RepotaxmannapiToken(
        query_text="SECTION 92C", org_text="section 92C",
        type="T1", or_in=False, proximity=0, query_date=None,
    )
    clauses = build_should_clauses([token], is_global=True, is_excus=False)

    minus_clauses = [
        c for c in clauses
        if "match_phrase" in c and "fullcontent" in c["match_phrase"]
        and c["match_phrase"]["fullcontent"]["query"] == "SUB SECTION 92C"
    ]
    assert len(minus_clauses) == 1
    assert minus_clauses[0]["match_phrase"]["fullcontent"]["boost"] == 1
