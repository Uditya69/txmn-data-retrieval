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


def _fullcontent_slop(clauses: list[dict], field: str = "fullcontent") -> int:
    """Pick out the plain (non-minus) fullcontent-tier clause's slop - i.e. the one
    whose query text is NOT the "SUB ..." minus-clause query."""
    matches = [
        c["match_phrase"][field] for c in clauses
        if "match_phrase" in c and field in c["match_phrase"]
        and not c["match_phrase"][field]["query"].startswith("SUB ")
    ]
    assert len(matches) == 1
    return matches[0]["slop"]


def test_fullcontent_slop_is_overridden_to_10000_for_a_tx_typed_token():
    # SearchTextElastic.cs:1104-1105: qt.QType == "TX" -> unconditional Slop(10000),
    # regardless of the token's actual proximity.
    token = RepotaxmannapiToken(
        query_text="Dimension Data India", org_text="Dimension Data India",
        type="TX", or_in=False, proximity=5, query_date=None,
    )
    clauses = build_should_clauses([token], is_global=True, is_excus=False)
    assert _fullcontent_slop(clauses) == 10000


def test_fullcontent_slop_is_overridden_to_10000_when_proximity_is_still_the_default():
    # SearchTextElastic.cs:1102-1103: for a non-TX token, slop is 10000 when
    # qt.QProximity still equals ProximityDefault.DefaultValue (5) - i.e. it was never
    # set to anything else.
    token = RepotaxmannapiToken(
        query_text="some text", org_text="some text",
        type="T1", or_in=False, proximity=5, query_date=None,
    )
    clauses = build_should_clauses([token], is_global=True, is_excus=False)
    assert _fullcontent_slop(clauses) == 10000


def test_fullcontent_slop_uses_actual_proximity_for_non_tx_non_default_proximity():
    # SearchTextElastic.cs:1102-1103: for a non-TX token whose proximity was explicitly
    # set away from the default, the actual proximity value is used as slop.
    token = RepotaxmannapiToken(
        query_text="some text", org_text="some text",
        type="T1", or_in=False, proximity=3, query_date=None,
    )
    clauses = build_should_clauses([token], is_global=True, is_excus=False)
    assert _fullcontent_slop(clauses) == 3


def test_fullcontent_slop_override_does_not_apply_in_the_excus_ph_branch():
    # SearchTextElastic.cs:1081 (Excus/PH branch's fullcontent.phrase_search line) uses
    # plain qt.QProximity with no override - unlike the non-Excus "else" branch's
    # fullcontent line at :1102-1105.
    token = RepotaxmannapiToken(
        query_text="Dimension Data India", org_text="Dimension Data India",
        type="TX", or_in=False, proximity=5, query_date=None,
    )
    clauses = build_should_clauses([token], is_global=True, is_excus=True)
    assert _fullcontent_slop(clauses, field="fullcontent.phrase_search") == 5
