"""Python port of repotaxmannapi/TaxmannAPI/Elastic/SearchTextElastic.cs's `GetQuery`
(a separate, read-only .NET codebase) - phrase-boost tiers, field names and the
SECTION-prefix exclusion rule, each verified 2026-09-02 by reading the real C# source
(not a summary of it) and cited by exact file:line below.

Scope note (read before extending): `GetQuery(TaxmannQueryAnalizer, Search)`
(SearchTextElastic.cs:787) is one large per-token switch with several branches this task
deliberately does NOT model, left for a later task:

- The pipe-separated ("|") OR-group branch (SearchTextElastic.cs:838-928, entered when
  `qt.QueryText.Split('|').Length > 1`) - explicitly out of scope per this task's brief
  ("OR-group/pipe-separated synonym handling").
- The `isGlobalSearch == "yes" && QType == "TX"` branch (SearchTextElastic.cs:1010-1042),
  which ORs together TWO boost tiers per field (e.g. heading 155000 at slop-4 OR heading
  90000 at plain slop) instead of the single flat tier this module emits. Also out of
  scope - it is itself an "OR of clauses" shape, same category of thing Task 5 owns.
- Date-token branches (MT/DM/DT) and citation assembly (CT) - explicitly out of scope
  per this task's brief.

This module models only the single-flat-tier-per-field shape: the plain "else" default
branch (SearchTextElastic.cs:1084-1107, hit for e.g. a bare TX token when the TX+global
special-case above does not apply) and its `isExcus`/PH counterpart
(SearchTextElastic.cs:1068-1082).

**Correction versus the task brief's sample values** (recorded here because the global
project rule requires every constant to be verbatim-cited, and this one did not check
out): the brief stated `fullcontent` boost as 5. Reading every `.Field(f => f.fullcontent)`
/ `Field("fullcontent.phrase_search")` call site in SearchTextElastic.cs (there are over a
dozen, across every branch: 886, 890, 909, 913, 964, 985, 1006, 1007, 1033, 1034, 1039,
1062, 1064, 1081, 1103, 1105) as well as the two advance-search helper methods
`GetAnyOfSearchGlobalSearchQuery` (line 573, 576) and `GetNotIncludeGlobalSearchQuery`
(line 618, 621), and the dead `OldGlobalSearchResearch.cs:74` - every single one passes
`Boost(1)`. There is no `Boost(5)` on `fullcontent` anywhere in the codebase. Using 5 would
have shipped a boost value that does not exist in the source being ported. Corrected to 1
here, with the citations above backing it.

**Note on the SECTION-prefix minus-clause condition**: in the real source this condition
(`qt.QType == "T1" && query.Trim().IndexOf("SECTION ") >= 0`, SearchTextElastic.cs:887 and
:910) only appears inside the pipe-split OR-group branch mentioned above (out of this
task's scope) - there is no standalone SECTION-minus-clause handling in the plain
single-token "else" branch this module otherwise mirrors. The task brief explicitly asks
for this rule to be ported into `build_should_clauses` regardless, matching the real
condition text (also present, boost 1, analyzer snowball, in the two advance-search helper
methods at lines 574/576 and 619/621) - so it is implemented here, generalized to apply to
any single T1-typed SECTION-prefixed token, not gated behind pipe-splitting (which is
Task 5's job to add on top).
"""
from __future__ import annotations

from common.repotaxmannapi_tokenizer import RepotaxmannapiToken

# Field -> boost, non-Excus (`.phrase_search` suffix, analyzer dropped) default "else"
# branch of GetQuery. Verbatim from SearchTextElastic.cs:1086-1103 (heading, subheading,
# searchboosttext, headnotestext, fullcontent respectively - headnotestext here uses only
# the primary 65000 tier at line 1094; the file also ORs in a secondary 50000 tier at
# line 1096 for non-"NZ" token types, which is out of scope - see module docstring).
_PHRASE_BOOSTS_STANDARD = {
    "heading": 155000,  # SearchTextElastic.cs:1086
    "subheading": 80000,  # SearchTextElastic.cs:1087
    "searchboosttext": 70000,  # SearchTextElastic.cs:1088
    "headnotestext": 65000,  # SearchTextElastic.cs:1094
    "fullcontent": 1,  # SearchTextElastic.cs:1103 (corrected from brief's stated 5 - see module docstring)
}

# Fields that carry `.Analyzer("snowball")` in the non-Excus branch (all of them, per
# SearchTextElastic.cs:1086-1103). Excus (PH) branch (SearchTextElastic.cs:1068-1081) never
# sets `.Analyzer(...)` at all - dropped entirely, not merely defaulted.
_SNOWBALL_ANALYZER_FIELDS = frozenset(_PHRASE_BOOSTS_STANDARD.keys())

# SUB-exclusion minus-clause boost, verbatim from SearchTextElastic.cs:890 and :913 (also
# GetAnyOfSearchGlobalSearchQuery:576, GetNotIncludeGlobalSearchQuery:621) - all Boost(1),
# same correction as _PHRASE_BOOSTS_STANDARD["fullcontent"] above.
_SUB_EXCLUSION_BOOST = 1


def build_should_clauses(
    tokens: list[RepotaxmannapiToken], is_global: bool, is_excus: bool
) -> list[dict]:
    """Build the `should`-clause phrase-boost tiers for a list of tokens, mirroring
    SearchTextElastic.cs's `GetQuery` default (non-OR-group) branch - see module
    docstring for exactly which parts of that method this covers and which it doesn't.

    `is_global` is accepted per this task's interface spec but not yet used - the
    branches that vary behavior by global-vs-non-global search (e.g. the TX+global
    dual-boost-tier branch at SearchTextElastic.cs:1010-1042) are out of scope for this
    task (see module docstring); it is threaded through so later tasks can extend this
    function without changing its signature.
    """
    del is_global  # accepted for interface stability; not yet used - see docstring
    should: list[dict] = []
    field_suffix = ".phrase_search" if is_excus else ""

    for token in tokens:
        for field, boost in _PHRASE_BOOSTS_STANDARD.items():
            field_name = f"{field}{field_suffix}"
            match_phrase: dict = {
                "query": token.query_text,
                "boost": boost,
                "slop": token.proximity,
            }
            if not is_excus and field in _SNOWBALL_ANALYZER_FIELDS:
                match_phrase["analyzer"] = "snowball"
            should.append({"match_phrase": {field_name: match_phrase}})

        # SECTION-prefix minus-clause: excludes documents where the section reference is
        # actually a sub-section back-reference. SearchTextElastic.cs:887-891, :910-914
        # (condition: `qt.QType == "T1" && query.Trim().IndexOf("SECTION ") >= 0`).
        if token.type == "T1" and "SECTION " in token.query_text:
            fullcontent_field = f"fullcontent{field_suffix}"
            minus_match_phrase: dict = {
                "query": f"SUB {token.query_text}",
                "boost": _SUB_EXCLUSION_BOOST,
                "slop": token.proximity,
            }
            if not is_excus:
                minus_match_phrase["analyzer"] = "snowball"
            should.append({"match_phrase": {fullcontent_field: minus_match_phrase}})

    return should
