"""Python port of repotaxmannapi/TaxmannAPI/Elastic/SearchTextElastic.cs's `GetQuery`
(a separate, read-only .NET codebase) - phrase-boost tiers, field names and the
SECTION-prefix exclusion rule, each verified 2026-09-02 by reading the real C# source
(not a summary of it) and cited by exact file:line below.

Scope note (read before extending): `GetQuery(TaxmannQueryAnalizer, Search)`
(SearchTextElastic.cs:787) is one large per-token switch with several branches. As of
Task 5, this module models:

- The plain "else" default branch (SearchTextElastic.cs:1084-1107) and its `isExcus`/PH
  counterpart (SearchTextElastic.cs:1068-1082) - Task 4.
- The pipe-separated ("|") OR-group branch (SearchTextElastic.cs:838-928, entered when
  `qt.QueryText.Split('|').Length > 1` - checked BEFORE the QType switch below, for ANY
  token type, per the real `if (q.Length > 1) {...} else { <QType switch> }` structure at
  842/929) - Task 5, see `_pipe_split_clauses`.
- The `isGlobalSearch == "yes" && QType == "TX"` dual/multi-boost-tier branch
  (SearchTextElastic.cs:1010-1042) - Task 5, see `_tx_global_clauses`.

Still explicitly out of scope (not modeled): date-token branches (MT/DM/DT,
SearchTextElastic.cs:948-1009) and citation assembly (CT, SearchTextElastic.cs:1043-1066).
Both introduce fields/inputs this function has no parameters for
(`documentdate` wildcard queries built from a month-name lookup table, `search.IsTopStory`,
`search.IsTldSearch`, `otherinfo.fullcitation`, `searchProcess.SearchTextOrg`,
`searchProcess.iGroupID`) - porting them soundly needs either new parameters threaded
through this function's signature (a design decision beyond "add a branch") or a
knowingly-incomplete port that silently drops real conditions, which the project's rules
explicitly forbid doing without disclosure. Left for a later task. Both were read in full
while tracing the two branches above (SearchTextElastic.cs:787-1116), not skipped.

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

**Note on the SECTION-prefix minus-clause condition**: the FULL real condition, verbatim,
is `qt.QType == "T1" && query.Trim().IndexOf("SECTION ") >= 0 && !search.isheadnoteToggle`
(SearchTextElastic.cs:887 and :910 - each is the `if` guarding the minus-clause build at
:889-890 and :912-913 respectively). That condition only appears inside the pipe-split
OR-group branch mentioned above (out of this task's scope) - there is no standalone
SECTION-minus-clause handling in the plain single-token "else" branch this module
otherwise mirrors. The task brief explicitly asks for this rule to be ported into
`build_should_clauses` regardless, matching the real condition text (also present, boost 1,
analyzer snowball, in the two advance-search helper methods at lines 574/576 and 619/621) -
so it is implemented here, generalized to apply to any single T1-typed SECTION-prefixed
token, not gated behind pipe-splitting (which is Task 5's job to add on top).

This implementation does NOT have access to `isheadnoteToggle` - `build_should_clauses`
has no `headnote_toggle` parameter, and threading one through is out of this task's scope.
So the third conjunct (`!search.isheadnoteToggle`) is simply not evaluated here: the
minus-clause fires whenever the first two conjuncts hold, unconditionally with respect to
that guard. This is a real, disclosed gap versus the exact real-source condition, not a
verbatim port of all three conjuncts.

**Note on the fullcontent slop override**: in the real source's "else" default branch,
the plain fullcontent tier (the one `_PHRASE_BOOSTS_STANDARD["fullcontent"]` models) does
NOT use the token's raw proximity as slop - it overrides it, per
SearchTextElastic.cs:1102-1105 (also present, identically, in the pipe-split OR-group
branch at :1061-1064 and :885-886/:908-909 - out of this task's scope):
```
if (qt.QType != "TX")
    ...Slop(TaxmannQueryAnalizer.ProximityDefault.DefaultValue == qt.QProximity ? 10000 : qt.QProximity));
else
    ...Slop(10000));
```
i.e.: a `TX`-typed token always gets slop 10000 for `fullcontent`; any other token gets
slop 10000 if its proximity equals `ProximityDefault.DefaultValue` (5, per
`TaxmannQueryAnalizer.cs:82`, ported here as `ProximityDefault.DEFAULT_VALUE` in
`repotaxmannapi_tokenizer.py`), else its actual proximity. `build_should_clauses` has
`token.type` and `token.proximity` in scope for every token already, so this is
implemented faithfully below rather than merely documented as a gap. Note this override
is specific to the `fullcontent` field/tier - the SUB-exclusion minus-clause built from the
same token also targets `fullcontent` but is a separate real-source statement
(SearchTextElastic.cs:890, :913) that uses plain `qt.QProximity` with no such override -
confirmed by reading those lines directly, so the minus-clause's slop is intentionally
left as plain `token.proximity` here.
"""
from __future__ import annotations

from common.repotaxmannapi_tokenizer import (
    ProximityDefault,
    RepotaxmannapiToken,
    TokenType,
)

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


def _tx_global_clauses(token: RepotaxmannapiToken) -> list[dict]:
    """`isGlobalSearch == "yes" && QType == "TX"` branch, SearchTextElastic.cs:1010-1042.
    ORs together several boost tiers per field (heading/subheading/searchboosttext: two
    tiers each at slop `proximity - 4` and plain `proximity`; headnotestext: three tiers,
    the third fixed at slop 100; fullcontent: two tiers when the query text contains a
    space, one otherwise) - modeled here as flat `should`-list entries per this module's
    existing convention (see the plain default branch above, which already flattens its
    own field tiers the same way rather than nesting AND/OR QueryContainer combinators).

    Not modeled (SearchTextElastic.cs:1021-1024): the `search.IsTopStory`/
    `search.IsTldSearch`-gated topstoryheading/tldheading tiers - this function has no
    such parameters, consistent with the rest of this module (the plain default branch
    above has the identical gap for the same fields at :1089-1092, pre-existing from
    Task 4). Also not modeled: the `!search.isheadnoteToggle` guard around the fullcontent
    tiers (:1029) - no such parameter exists here either (same documented gap as the
    SECTION-minus-clause note below), so the fullcontent tiers are always emitted.
    """
    query = token.query_text  # normalization at SearchTextElastic.cs:935-947 not ported - see module docstring
    should: list[dict] = []

    def _mp(field: str, boost: int, slop: int) -> dict:
        return {
            "match_phrase": {
                field: {"query": query, "boost": boost, "slop": slop, "analyzer": "snowball"}
            }
        }

    # heading: SearchTextElastic.cs:1012-1014
    should.append(_mp("heading", 155000, token.proximity - 4))
    should.append(_mp("heading", 90000, token.proximity))
    # subheading: SearchTextElastic.cs:1015-1017
    should.append(_mp("subheading", 80000, token.proximity - 4))
    should.append(_mp("subheading", 75000, token.proximity))
    # searchboosttext: SearchTextElastic.cs:1018-1020 (uses querySearchboosttext, which
    # equals `query` here - see the normalization note above)
    should.append(_mp("searchboosttext", 70000, token.proximity - 4))
    should.append(_mp("searchboosttext", 67000, token.proximity))
    # headnotestext: SearchTextElastic.cs:1025-1028 (three tiers - Headnotes3's slop is
    # the fixed literal 100, not proximity-derived)
    should.append(_mp("headnotestext", 65000, token.proximity - 4))
    should.append(_mp("headnotestext", 60000, token.proximity))
    should.append(_mp("headnotestext", 50000, 100))
    # fullcontent: SearchTextElastic.cs:1029-1041
    if " " in query:
        should.append(_mp("fullcontent", 100, token.proximity))
        should.append(_mp("fullcontent", 1, 5000))
    else:
        should.append(_mp("fullcontent", 1, 5000))

    return should


def _pipe_split_clauses(token: RepotaxmannapiToken) -> list[dict]:
    """Pipe-separated ("|") OR-group branch, SearchTextElastic.cs:838-928. Splits
    `qt.QueryText` on '|' (verbatim `Split('|')` - no per-alternative trimming, so an
    alternative may keep leading/trailing whitespace from how it was joined, e.g. by
    `TaxmannQueryAnalizer._process_key_word`'s `" | "`-separated joins) and builds the
    same field-tier clauses for EACH alternative, mirroring the identical clause
    construction present (verbatim, just combined via `|=` vs `&=`) in both the
    `if (isOver)` (877-895) and `else` (900-918) halves of the per-alternative
    `ForEach` body - this function treats every alternative uniformly, per this module's
    flat `should`-list convention (see `build_should_clauses`'s docstring on precedence).

    Not modeled: the cross-token `isOver`/`orCounter` stateful bookkeeping that groups
    each pipe-token's alternatives into their own `QueryContainer` OR-group and commits it
    to a list once the NEXT pipe-token starts (838-840, 861-876, 892-895, 915-919 in part)
    - that machinery exists to build nested AND-of-OR `QueryContainer` trees for the
    Elasticsearch client library used by the C# (NEST), a shape this module's flat
    `should`-list return type does not represent at all (nor does the plain default
    branch this module already ported in Task 4, which also flattens away its own
    `&=`/`AND` field combination). Also not modeled: `search.IsTopStory`/
    `search.IsTldSearch`-gated tiers (880-883/903-906) and the `!search.isheadnoteToggle`
    guard around the fullcontent tier (885/908) and the `searchheadingnumber`
    `else if (qt.QType == "T1" && searchProcess.iGroupID == Constants_GetIdByName.CirNot)`
    tier (892-894/915-917) - all three need parameters this function does not have,
    same documented-gap pattern as elsewhere in this module. The `taxmann com`/`compcase`
    query-text normalization (848-860) is likewise not ported - pre-existing gap, the
    plain default branch never ported it either (see its own comment at query-text use).
    """
    should: list[dict] = []
    for alt in token.query_text.split("|"):  # SearchTextElastic.cs:841
        query = alt  # normalization at :848-860 not ported - see docstring above
        should.append(
            {"match_phrase": {"heading": {
                "query": query, "boost": 155000, "slop": token.proximity, "analyzer": "snowball",
            }}}
        )  # SearchTextElastic.cs:877/900
        should.append(
            {"match_phrase": {"subheading": {
                "query": query, "boost": 80000, "slop": token.proximity, "analyzer": "snowball",
            }}}
        )  # SearchTextElastic.cs:878/901
        should.append(
            {"match_phrase": {"searchboosttext": {
                "query": query, "boost": 70000, "slop": token.proximity, "analyzer": "snowball",
            }}}
        )  # SearchTextElastic.cs:879/902 (querySearchboosttext == query - see docstring)
        should.append(
            {"match_phrase": {"headnotestext": {
                "query": query, "boost": 65000, "slop": token.proximity, "analyzer": "snowball",
            }}}
        )  # SearchTextElastic.cs:884/907

        # fullcontent slop override, SearchTextElastic.cs:885-886/908-909: NO `qt.QType !=
        # "TX"` guard here (unlike the default branch's :1102-1105) - always 10000 exactly
        # when proximity is still the untouched default, else the actual proximity,
        # regardless of token type.
        fc_slop = 10000 if token.proximity == ProximityDefault.DEFAULT_VALUE else token.proximity
        should.append(
            {"match_phrase": {"fullcontent": {
                "query": query, "boost": 1, "slop": fc_slop, "analyzer": "snowball",
            }}}
        )  # SearchTextElastic.cs:885-886/908-909

        # SECTION-prefix minus-clause, SearchTextElastic.cs:887-891/910-914. FULL real
        # condition, verbatim: `qt.QType == "T1" && query.Trim().IndexOf("SECTION ") >= 0
        # && !search.isheadnoteToggle` - third conjunct not evaluated here, same disclosed
        # gap as build_should_clauses's own SECTION-minus-clause block below.
        if token.type == "T1" and "SECTION " in query.strip():
            should.append(
                {"match_phrase": {"fullcontent": {
                    "query": f"SUB {query}", "boost": 1, "slop": token.proximity,
                    "analyzer": "snowball",
                }}}
            )  # SearchTextElastic.cs:890/913

    return should


def build_should_clauses(
    tokens: list[RepotaxmannapiToken], is_global: bool, is_excus: bool
) -> list[dict]:
    """Build the `should`-clause phrase-boost tiers for a list of tokens, mirroring
    SearchTextElastic.cs's `GetQuery` - see module docstring for exactly which parts of
    that method this covers and which it doesn't.

    Per-token branch precedence mirrors the real `if (q.Length > 1) {...} else {...}`
    structure at SearchTextElastic.cs:842/929 followed by the QType `if`/`else if` chain
    inside the `else` (948-1107): a pipe-separated (`|`) token is dispatched to
    `_pipe_split_clauses` REGARDLESS of its type (the pipe check comes first, before any
    QType check, for every token) - see `_pipe_split_clauses`. Otherwise, a `TX`-typed
    token under `is_global=True` is dispatched to `_tx_global_clauses`
    (SearchTextElastic.cs:1010, `searchProcess.isGlobalSearch == "yes" && qt.QType ==
    "TX"`, checked ahead of the "CT"/"PH"/else branches in the same chain) - UNLESS
    `is_excus` is set, in which case it is skipped in favor of the plain `isExcus`/PH
    logic below. This `is_excus`-first ordering is a port-level convention, not a literal
    mirror of real branch order: `is_excus` in this module (per Task 4) already stands for
    "render the PH/`.phrase_search` variant of this call" independent of any single
    token's real `QType` (Task 4's own tests exercise it against a `TX`-typed token), so
    there is no real-source scenario of a token being simultaneously `QType == "TX"` and
    `QType == "PH"` for this ordering decision to contradict - both real branches are
    mutually exclusive on `QType` and neither's clause construction has anything to do
    with the other. Any token that is neither pipe-separated nor (TX-typed AND global AND
    not-excus) falls through to the plain "else"/`isExcus` logic below, unchanged from
    Task 4.
    """
    should: list[dict] = []
    field_suffix = ".phrase_search" if is_excus else ""

    for token in tokens:
        if "|" in token.query_text:
            # SearchTextElastic.cs:841-842: `qt.QueryText.Split('|')` - checked before
            # any QType branch, for every token type.
            should.extend(_pipe_split_clauses(token))
            continue
        if is_global and not is_excus and token.type == TokenType.TEXT:
            # SearchTextElastic.cs:1010: `searchProcess.isGlobalSearch == "yes" &&
            # qt.QType == "TX"`. `not is_excus` guard: see docstring above.
            should.extend(_tx_global_clauses(token))
            continue

        for field, boost in _PHRASE_BOOSTS_STANDARD.items():
            field_name = f"{field}{field_suffix}"
            if field == "fullcontent" and not is_excus:
                # fullcontent slop override, SearchTextElastic.cs:1102-1105 (see module
                # docstring "Note on the fullcontent slop override"): a TX-typed token
                # always gets slop 10000; any other token gets 10000 if its proximity is
                # still the untouched default (5), else its actual proximity. This
                # override is only present in the non-Excus "else" branch this module
                # otherwise models - the Excus (PH) branch's fullcontent line
                # (SearchTextElastic.cs:1081) uses plain qt.QProximity with no override,
                # confirmed by reading that line directly.
                if token.type == TokenType.TEXT:
                    slop = 10000
                elif token.proximity == ProximityDefault.DEFAULT_VALUE:
                    slop = 10000
                else:
                    slop = token.proximity
            else:
                slop = token.proximity
            match_phrase: dict = {
                "query": token.query_text,
                "boost": boost,
                "slop": slop,
            }
            if not is_excus and field in _SNOWBALL_ANALYZER_FIELDS:
                match_phrase["analyzer"] = "snowball"
            should.append({"match_phrase": {field_name: match_phrase}})

        # SECTION-prefix minus-clause: excludes documents where the section reference is
        # actually a sub-section back-reference. SearchTextElastic.cs:887-891, :910-914.
        # FULL real condition, verbatim: `qt.QType == "T1" && query.Trim().IndexOf
        # ("SECTION ") >= 0 && !search.isheadnoteToggle` - this implementation does not
        # have access to `isheadnoteToggle` (no such parameter here) and so applies the
        # rule below unconditionally with respect to that third conjunct; see module
        # docstring "Note on the SECTION-prefix minus-clause condition" for the full
        # disclosure.
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
