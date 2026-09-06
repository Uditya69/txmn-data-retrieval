from common.repotaxmannapi_query_builder import build_should_clauses
from common.repotaxmannapi_tokenizer import RepotaxmannapiToken


def _flatten_or_groups(result: dict[str, list[list[dict]]], field: str) -> list[dict]:
    """Test helper: flatten one field's list of OR-groups back into a single flat list of
    match_phrase clause dicts, for tests that only care about which clauses exist for a
    field, not the AND/OR grouping itself."""
    return [clause for or_group in result.get(field, []) for clause in or_group]


def test_builds_phrase_boost_should_clauses_for_a_plain_text_token():
    # is_global=False here (not True as originally, pre-Task-5): now that the TX+global
    # dual-boost-tier branch (SearchTextElastic.cs:1010-1042) is implemented, a TX-typed
    # token under is_global=True takes THAT branch, not this plain default one - see
    # test_tx_global_branch_* below. A TX token only reaches this plain default branch
    # when isGlobalSearch != "yes", i.e. is_global=False here.
    token = RepotaxmannapiToken(
        query_text="Dimension Data India", org_text="Dimension Data India",
        type="TX", or_in=False, proximity=5, query_date=None,
    )
    result = build_should_clauses([token], is_global=False, is_excus=False)

    boosts_by_field = {
        field: _flatten_or_groups(result, field)[0]["match_phrase"][field]["boost"]
        for field in ("heading", "subheading", "searchboosttext", "headnotes_text", "fullcontent")
    }
    assert boosts_by_field == {
        "heading": 155000, "subheading": 80000,
        "searchboosttext": 70000, "headnotes_text": 65000, "fullcontent": 1,
    }


def test_adds_sub_exclusion_clause_for_a_section_prefixed_token():
    token = RepotaxmannapiToken(
        query_text="SECTION 92C", org_text="section 92C",
        type="T1", or_in=False, proximity=0, query_date=None,
    )
    result = build_should_clauses([token], is_global=True, is_excus=False)

    minus_clauses = [
        c for c in _flatten_or_groups(result, "_fullcontent_minus")
        if c["match_phrase"]["fullcontent"]["query"] == "SUB SECTION 92C"
    ]
    assert len(minus_clauses) == 1
    assert minus_clauses[0]["match_phrase"]["fullcontent"]["boost"] == 1


def _fullcontent_slop(result: dict[str, list[list[dict]]], field: str = "fullcontent") -> int:
    """Pick out the plain (non-minus) fullcontent-tier clause's slop - i.e. the one
    whose query text is NOT the "SUB ..." minus-clause query."""
    matches = [
        c["match_phrase"][field] for c in _flatten_or_groups(result, field)
        if not c["match_phrase"][field]["query"].startswith("SUB ")
    ]
    assert len(matches) == 1
    return matches[0]["slop"]


def test_fullcontent_slop_is_overridden_to_10000_for_a_tx_typed_token():
    # SearchTextElastic.cs:1104-1105: qt.QType == "TX" -> unconditional Slop(10000),
    # regardless of the token's actual proximity. is_global=False (see the note in
    # test_builds_phrase_boost_should_clauses_for_a_plain_text_token above) - this line
    # only runs for a TX token when isGlobalSearch != "yes".
    token = RepotaxmannapiToken(
        query_text="Dimension Data India", org_text="Dimension Data India",
        type="TX", or_in=False, proximity=5, query_date=None,
    )
    result = build_should_clauses([token], is_global=False, is_excus=False)
    assert _fullcontent_slop(result) == 10000


def test_fullcontent_slop_is_overridden_to_10000_when_proximity_is_still_the_default():
    # SearchTextElastic.cs:1102-1103: for a non-TX token, slop is 10000 when
    # qt.QProximity still equals ProximityDefault.DefaultValue (5) - i.e. it was never
    # set to anything else.
    token = RepotaxmannapiToken(
        query_text="some text", org_text="some text",
        type="T1", or_in=False, proximity=5, query_date=None,
    )
    result = build_should_clauses([token], is_global=True, is_excus=False)
    assert _fullcontent_slop(result) == 10000


def test_fullcontent_slop_uses_actual_proximity_for_non_tx_non_default_proximity():
    # SearchTextElastic.cs:1102-1103: for a non-TX token whose proximity was explicitly
    # set away from the default, the actual proximity value is used as slop.
    token = RepotaxmannapiToken(
        query_text="some text", org_text="some text",
        type="T1", or_in=False, proximity=3, query_date=None,
    )
    result = build_should_clauses([token], is_global=True, is_excus=False)
    assert _fullcontent_slop(result) == 3


def test_fullcontent_slop_override_does_not_apply_in_the_excus_ph_branch():
    # SearchTextElastic.cs:1081 (Excus/PH branch's fullcontent.phrase_search line) uses
    # plain qt.QProximity with no override - unlike the non-Excus "else" branch's
    # fullcontent line at :1102-1105.
    token = RepotaxmannapiToken(
        query_text="Dimension Data India", org_text="Dimension Data India",
        type="TX", or_in=False, proximity=5, query_date=None,
    )
    result = build_should_clauses([token], is_global=True, is_excus=True)
    assert _fullcontent_slop(result, field="fullcontent.phrase_search") == 5


def _match_phrase_tuples(result: dict[str, list[list[dict]]], field: str) -> list[tuple]:
    """All (query, boost, slop) triples for match_phrase clauses on `field`, in order."""
    return [
        (c["match_phrase"][field]["query"], c["match_phrase"][field]["boost"],
         c["match_phrase"][field]["slop"])
        for c in _flatten_or_groups(result, field)
    ]


# ---------------------------------------------------------------------------------------
# TX-type + is_global dual/multi-boost-tier branch - SearchTextElastic.cs:1010-1042.
# ---------------------------------------------------------------------------------------

def test_tx_global_branch_ors_two_heading_tiers_at_proximity_minus_4_and_proximity():
    # SearchTextElastic.cs:1012-1014: heading1 Boost(155000).Slop(qt.QProximity - 4) OR
    # heading2 Boost(90000).Slop(qt.QProximity).
    token = RepotaxmannapiToken(
        query_text="Dimension Data India", org_text="Dimension Data India",
        type="TX", or_in=False, proximity=5, query_date=None,
    )
    result = build_should_clauses([token], is_global=True, is_excus=False)
    tuples = _match_phrase_tuples(result, "heading")
    assert ("Dimension Data India", 155000, 1) in tuples  # 5 - 4 = 1
    assert ("Dimension Data India", 90000, 5) in tuples
    # Both tiers come from the SAME token's OR-group - exactly one OR-group for this field.
    assert len(result["heading"]) == 1


def test_tx_global_branch_ors_two_subheading_and_searchboosttext_tiers():
    # SearchTextElastic.cs:1015-1017 (subheading 80000/75000), :1018-1020
    # (searchboosttext 70000/67000).
    token = RepotaxmannapiToken(
        query_text="Dimension Data India", org_text="Dimension Data India",
        type="TX", or_in=False, proximity=5, query_date=None,
    )
    result = build_should_clauses([token], is_global=True, is_excus=False)
    sub_tuples = _match_phrase_tuples(result, "subheading")
    assert ("Dimension Data India", 80000, 1) in sub_tuples
    assert ("Dimension Data India", 75000, 5) in sub_tuples
    sbt_tuples = _match_phrase_tuples(result, "searchboosttext")
    assert ("Dimension Data India", 70000, 1) in sbt_tuples
    assert ("Dimension Data India", 67000, 5) in sbt_tuples


def test_tx_global_branch_ors_three_headnotestext_tiers():
    # SearchTextElastic.cs:1025-1028: Headnotes1 (65000, proximity-4), Headnotes2 (60000,
    # proximity), Headnotes3 (50000, slop 100 - fixed, not proximity-derived).
    token = RepotaxmannapiToken(
        query_text="Dimension Data India", org_text="Dimension Data India",
        type="TX", or_in=False, proximity=5, query_date=None,
    )
    result = build_should_clauses([token], is_global=True, is_excus=False)
    tuples = _match_phrase_tuples(result, "headnotes_text")
    assert ("Dimension Data India", 65000, 1) in tuples
    assert ("Dimension Data India", 60000, 5) in tuples
    assert ("Dimension Data India", 50000, 100) in tuples


def test_tx_global_branch_fullcontent_has_two_tiers_when_query_contains_a_space():
    # SearchTextElastic.cs:1031-1036: query.IndexOf(" ") >= 0 -> Fullcontent1
    # (boost 100, slop qt.QProximity) OR Fullcontent2 (boost 1, slop 5000).
    token = RepotaxmannapiToken(
        query_text="Dimension Data India", org_text="Dimension Data India",
        type="TX", or_in=False, proximity=5, query_date=None,
    )
    result = build_should_clauses([token], is_global=True, is_excus=False)
    tuples = _match_phrase_tuples(result, "fullcontent")
    assert ("Dimension Data India", 100, 5) in tuples
    assert ("Dimension Data India", 1, 5000) in tuples
    assert len(tuples) == 2
    # Both tiers form ONE OR-group (one token's own alternatives).
    assert len(result["fullcontent"]) == 1


def test_tx_global_branch_fullcontent_is_single_tier_when_query_has_no_space():
    # SearchTextElastic.cs:1037-1039: else branch (no space in query) -> a single
    # Boost(1).Slop(5000) clause, no Boost(100) tier.
    token = RepotaxmannapiToken(
        query_text="Infosys", org_text="Infosys",
        type="TX", or_in=False, proximity=5, query_date=None,
    )
    result = build_should_clauses([token], is_global=True, is_excus=False)
    tuples = _match_phrase_tuples(result, "fullcontent")
    assert tuples == [("Infosys", 1, 5000)]


def test_tx_global_branch_not_used_when_is_global_is_false():
    # searchProcess.isGlobalSearch == "yes" is required; a non-global TX token falls
    # through to the plain default "else" branch (SearchTextElastic.cs:1084-1107), which
    # this module already models (single heading tier at boost 155000).
    token = RepotaxmannapiToken(
        query_text="Dimension Data India", org_text="Dimension Data India",
        type="TX", or_in=False, proximity=5, query_date=None,
    )
    result = build_should_clauses([token], is_global=False, is_excus=False)
    tuples = _match_phrase_tuples(result, "heading")
    # heading has no slop override in the default branch (only fullcontent does) - plain
    # qt.QProximity (5), SearchTextElastic.cs:1086.
    assert tuples == [("Dimension Data India", 155000, 5)]
    # No 90000-boost second tier from the TX-global branch.
    assert all(boost != 90000 for _, boost, _ in tuples)


# ---------------------------------------------------------------------------------------
# Pipe-separated ("|") OR-group branch - SearchTextElastic.cs:838-928.
# ---------------------------------------------------------------------------------------

def test_pipe_split_token_builds_one_heading_clause_per_alternative():
    # SearchTextElastic.cs:841 (q = qt.QueryText.Split('|')), :877/:900 (heading clause
    # built per alternative, Boost(155000), Slop(qt.QProximity) - unmodified proximity,
    # unlike the TX-global branch above).
    token = RepotaxmannapiToken(
        query_text="92 | 092", org_text="92 | 092",
        type="T1", or_in=False, proximity=0, query_date=None,
    )
    result = build_should_clauses([token], is_global=True, is_excus=False)
    tuples = _match_phrase_tuples(result, "heading")
    # verbatim Split('|') - no per-alt trim in the C#, so the first alt keeps its
    # trailing space and the second its leading space.
    assert ("92 ", 155000, 0) in tuples
    assert (" 092", 155000, 0) in tuples
    # The two pipe-alternatives form ONE OR-group (one token's own alternatives).
    assert len(result["heading"]) == 1


def test_pipe_split_token_searchboosttext_and_subheading_per_alternative():
    # SearchTextElastic.cs:878/:901 (subheading 80000), :879/:902 (searchboosttext 70000,
    # built from querySearchboosttext not query - but for an alt with no "taxmann.com"
    # substring the two are equal).
    token = RepotaxmannapiToken(
        query_text="92 | 092", org_text="92 | 092",
        type="T1", or_in=False, proximity=0, query_date=None,
    )
    result = build_should_clauses([token], is_global=True, is_excus=False)
    assert _match_phrase_tuples(result, "subheading") == [("92 ", 80000, 0), (" 092", 80000, 0)]
    assert _match_phrase_tuples(result, "searchboosttext") == [("92 ", 70000, 0), (" 092", 70000, 0)]


def test_pipe_split_token_fullcontent_slop_override_ignores_qtype():
    # SearchTextElastic.cs:885-886/:908-909: the pipe-branch's fullcontent override is
    # `ProximityDefault.DefaultValue == qt.QProximity ? 10000 : qt.QProximity` with NO
    # `qt.QType != "TX"` guard (unlike the default "else" branch's :1102-1105) - so even a
    # TX-typed token here uses actual proximity when it's non-default.
    token = RepotaxmannapiToken(
        query_text="a | b", org_text="a | b",
        type="TX", or_in=False, proximity=3, query_date=None,
    )
    result = build_should_clauses([token], is_global=True, is_excus=False)
    tuples = _match_phrase_tuples(result, "fullcontent")
    assert ("a ", 1, 3) in tuples  # verbatim Split('|') - no per-alt trim in the C#
    assert (" b", 1, 3) in tuples


def test_pipe_split_token_fullcontent_slop_is_10000_at_default_proximity():
    token = RepotaxmannapiToken(
        query_text="a | b", org_text="a | b",
        type="TX", or_in=False, proximity=5, query_date=None,
    )
    result = build_should_clauses([token], is_global=True, is_excus=False)
    tuples = _match_phrase_tuples(result, "fullcontent")
    assert ("a ", 1, 10000) in tuples  # verbatim Split('|') - no per-alt trim in the C#
    assert (" b", 1, 10000) in tuples


def test_pipe_split_token_adds_section_minus_clause_per_matching_alternative():
    # SearchTextElastic.cs:887-891/:910-914: each alternative independently checked for
    # T1 type + "SECTION " prefix; minus-clause built per matching alt.
    token = RepotaxmannapiToken(
        query_text="SECTION 92C | SECTION 092C", org_text="section 92C",
        type="T1", or_in=False, proximity=0, query_date=None,
    )
    result = build_should_clauses([token], is_global=True, is_excus=False)
    minus_queries = {
        c["match_phrase"]["fullcontent"]["query"]
        for c in _flatten_or_groups(result, "_fullcontent_minus")
    }
    # verbatim Split('|') - no per-alt trim, so "SUB " + query keeps the split's own
    # leading/trailing whitespace (a trailing space on the first alt, a leading space -
    # hence the double space - on the second).
    assert minus_queries == {"SUB SECTION 92C ", "SUB  SECTION 092C"}


_CIRNOT_GROUP_ID = "111050000000000057"


def test_pipe_split_token_adds_searchheadingnumber_clause_for_cirnot_group_without_section_prefix():
    # SearchTextElastic.cs:892-894/915-917: `else if (qt.QType == "T1" &&
    # searchProcess.iGroupID == Constants_GetIdByName.CirNot)` - mutually exclusive with
    # the SECTION-minus-clause, fires when the alt is T1-typed, NOT "SECTION "-prefixed,
    # and the query's resolved group_id is CirNot.
    token = RepotaxmannapiToken(
        query_text="92 | 092", org_text="92 | 092",
        type="T1", or_in=False, proximity=3, query_date=None,
    )
    result = build_should_clauses(
        [token], is_global=True, is_excus=False, group_id=_CIRNOT_GROUP_ID,
    )
    tuples = _match_phrase_tuples(result, "searchheadingnumber")
    assert ("92 ", 85000, 3) in tuples
    assert (" 092", 85000, 3) in tuples


def test_pipe_split_token_no_searchheadingnumber_clause_when_group_id_is_not_cirnot():
    token = RepotaxmannapiToken(
        query_text="92 | 092", org_text="92 | 092",
        type="T1", or_in=False, proximity=3, query_date=None,
    )
    result = build_should_clauses([token], is_global=True, is_excus=False, group_id="0")
    assert "searchheadingnumber" not in result


def test_pipe_split_token_section_minus_clause_takes_precedence_over_searchheadingnumber():
    # The two branches are `if`/`else if` in the real source - a SECTION-prefixed alt gets
    # the minus-clause, never the searchheadingnumber tier, even under a CirNot group_id.
    token = RepotaxmannapiToken(
        query_text="SECTION 92C", org_text="section 92C",
        type="T1", or_in=False, proximity=0, query_date=None,
    )
    result = build_should_clauses(
        [token], is_global=True, is_excus=False, group_id=_CIRNOT_GROUP_ID,
    )
    assert "searchheadingnumber" not in result
    assert any(
        c["match_phrase"]["fullcontent"]["query"] == "SUB SECTION 92C"
        for c in _flatten_or_groups(result, "_fullcontent_minus")
    )


def test_citation_token_builds_the_five_field_tiers_plus_fullcontent():
    # SearchTextElastic.cs:1043-1066: CT branch. heading/subheading/searchboosttext at
    # their standard boosts, ONE headnotestext tier (no secondary 50000 tier, unlike the
    # plain default branch), the otherinfo.fullcitation.name cross-reference tier at
    # boost 155000 (same as heading), and the fullcontent tier.
    token = RepotaxmannapiToken(
        query_text="571 Ahd 2016", org_text="571/Ahd/2016",
        type="CT", or_in=False, proximity=2, query_date=None,
    )
    result = build_should_clauses([token], is_global=True, is_excus=False)

    boosts_by_field = {
        field: _flatten_or_groups(result, field)[0]["match_phrase"][field]["boost"]
        for field in (
            "heading", "subheading", "searchboosttext", "headnotes_text",
            "otherinfo.fullcitation.name", "fullcontent",
        )
    }
    assert boosts_by_field == {
        "heading": 155000, "subheading": 80000, "searchboosttext": 70000,
        "headnotes_text": 65000, "otherinfo.fullcitation.name": 155000, "fullcontent": 1,
    }


def test_citation_token_fullcitation_clause_uses_query_proximity_as_slop():
    token = RepotaxmannapiToken(
        query_text="571 Ahd 2016", org_text="571/Ahd/2016",
        type="CT", or_in=False, proximity=2, query_date=None,
    )
    result = build_should_clauses([token], is_global=True, is_excus=False)

    citation_clause = _flatten_or_groups(result, "otherinfo.fullcitation.name")[0]["match_phrase"][
        "otherinfo.fullcitation.name"
    ]
    assert citation_clause == {
        "query": "571 Ahd 2016", "boost": 155000, "slop": 2, "analyzer": "snowball",
    }


def test_citation_token_ignores_is_global_and_is_excus_gates():
    # SearchTextElastic.cs:1043: `qt.QType == "CT"` is its own unconditional branch, not
    # gated behind isGlobalSearch or the PH/Excus rendering - a CT token takes the same
    # branch regardless of is_global/is_excus.
    token = RepotaxmannapiToken(
        query_text="571 Ahd 2016", org_text="571/Ahd/2016",
        type="CT", or_in=False, proximity=2, query_date=None,
    )
    result_global = build_should_clauses([token], is_global=True, is_excus=False)
    result_not_global = build_should_clauses([token], is_global=False, is_excus=False)
    assert result_global == result_not_global


def test_default_branch_headnotestext_has_a_secondary_50000_boost_tier_at_slop_100():
    # SearchTextElastic.cs:1093-1097: Headnotes1 (65000, qt.QProximity) OR Headnotes2
    # (50000, slop 100) for every token type except NUM_ALPHA_ZONE ("NZ").
    token = RepotaxmannapiToken(
        query_text="Dimension Data India", org_text="Dimension Data India",
        type="TX", or_in=False, proximity=5, query_date=None,
    )
    result = build_should_clauses([token], is_global=False, is_excus=False)
    or_group = result["headnotes_text"][0]
    boosts_and_slops = {
        (c["match_phrase"]["headnotes_text"]["boost"], c["match_phrase"]["headnotes_text"]["slop"])
        for c in or_group
    }
    assert (65000, 5) in boosts_and_slops
    assert (50000, 100) in boosts_and_slops


def test_default_branch_headnotestext_secondary_tier_absent_for_num_alpha_zone_type():
    token = RepotaxmannapiToken(
        query_text="XYZ123", org_text="XYZ123",
        type="NZ", or_in=False, proximity=5, query_date=None,
    )
    result = build_should_clauses([token], is_global=False, is_excus=False)
    or_group = result["headnotes_text"][0]
    assert len(or_group) == 1
    assert or_group[0]["match_phrase"]["headnotes_text"]["boost"] == 65000


# ---------------------------------------------------------------------------------------
# "taxmann com"/"compcase" query-text normalization - SearchTextElastic.cs:848-860 /
# :935-947.
# ---------------------------------------------------------------------------------------

def test_taxmann_com_normalized_to_taxmann_dot_com_for_heading_field():
    # SearchTextElastic.cs:848-850 (pipe-split) / :935-937 (non-pipe branches, same logic)
    token = RepotaxmannapiToken(
        query_text="taxmann com 123", org_text="taxmann.com 123",
        type="TX", or_in=False, proximity=5, query_date=None,
    )
    result = build_should_clauses([token], is_global=False, is_excus=False)
    heading_clause = result["heading"][0][0]
    assert heading_clause["match_phrase"]["heading"]["query"] == "taxmann.com 123"


def test_taxmann_dot_com_normalized_to_taxmann_com_for_searchboosttext_field_only():
    # SearchTextElastic.cs:860: querySearchboosttext goes the OPPOSITE direction from
    # `query` - only searchboosttext gets this reverse rewrite.
    token = RepotaxmannapiToken(
        query_text="taxmann.com 123", org_text="taxmann.com 123",
        type="TX", or_in=False, proximity=5, query_date=None,
    )
    result = build_should_clauses([token], is_global=False, is_excus=False)
    searchboosttext_clause = result["searchboosttext"][0][0]
    heading_clause = result["heading"][0][0]
    assert searchboosttext_clause["match_phrase"]["searchboosttext"]["query"] == "taxmann com 123"
    assert heading_clause["match_phrase"]["heading"]["query"] == "taxmann.com 123"  # unchanged - no "taxmann com" substring present


def test_compcase_normalized_to_comp_case():
    # SearchTextElastic.cs:852-854
    token = RepotaxmannapiToken(
        query_text="compcase 45", org_text="compcase 45",
        type="TX", or_in=False, proximity=5, query_date=None,
    )
    result = build_should_clauses([token], is_global=False, is_excus=False)
    heading_clause = result["heading"][0][0]
    assert heading_clause["match_phrase"]["heading"]["query"] == "comp case 45"
