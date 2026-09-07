
# repotaxmannapi Query-Builder Exact-Parity Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the remaining gaps between this repo's Python port of `repotaxmannapi`'s
(a separate, read-only .NET codebase) query-builder logic and the real source, found by a
2026-09-06 three-way audit (`GetQuery` branches, FunctionScore/filters, tokenizer). Three of
the gaps are real correctness bugs (wrong ranking/results today); the rest are missing query
logic the real system has and ours doesn't. UI-toggle-gated logic (topstoryheading,
tldheading, isheadnoteToggle, subject-query, advance-search, filter-panel, date-sort,
paging clamp) is explicitly out of scope per the user's own instruction ("no UI things, no
filters, only ES query logic for normal + exact phrase searches").

**Architecture:** The core structural fix (Task 2) changes `build_should_clauses`'s return
shape from one flat `should`-list to a per-field-tier structure that preserves the real
source's AND-across-tokens / OR-across-alternatives semantics, then changes
`es_client.py`'s assembly to combine those per-field-tier results correctly. Every other
task is either a value correction (quick) or an additive new branch (query-text
normalization, date-token branches, two new should-clauses, a tokenizer-level rewrite
pass) that plugs into the same per-field-tier structure Task 2 establishes. Do the tasks
in the order listed — later tasks depend on Task 2's shape.

**Tech Stack:** Python 3.11, pytest + pytest-asyncio (existing test stack), no new
dependencies.

**Spec:** This plan's own "Background" sections (below) carry the full real-source
citations in place of a separate spec doc — this is a bugfix/gap-closure plan against
existing, already-specified behavior (`docs/superpowers/specs/2026-09-01-repotaxmannapi-exact-replica-design.md`),
not new feature design.

## Global Constraints

- Every constant (boost weight, factor, modifier, slop) must be copied verbatim from the
  real C# source with an exact file:line citation in the code comment — never approximated.
- `repotaxmannapi` (at `C:\A_VANSH\ALL_NEEDS\CODING\Taxmann\txmn-data-retrieval\misc\repotaxmannapi`)
  is read-only, uncompilable from this environment — verification is by careful reading, not
  by running the original C# code.
- No UI-toggle-gated logic (topstoryheading/tldheading/isheadnoteToggle/subject-text/
  advance-search/filter-panel/date-sort/paging-clamp) — explicitly out of scope, confirmed
  with the user.
- Every task ends with `uv run pytest packages/common/tests -q` passing (this repo's whole
  suite for the `common` package is fast — a few seconds — so no need to scope narrower)
  before commit. One pre-existing, unrelated failure
  (`test_build_query_preview_matches_what_raw_search_actually_sends`, a classifier-threshold
  issue predating this plan) is expected and must remain the ONLY failure after every task.
- Work happens in `C:\A_VANSH\ALL_NEEDS\CODING\Taxmann\txmn-data-retrieval\backend` (the
  split-out backend repo, origin `ai-intent-search.git`) — not the old monorepo path.

---

## Task 1: Correct static group-membership boost values to the real, unscaled weights

**Files:**
- Modify: `packages/common/src/common/data/repotaxmannapi_boost_config.json:45-61`
  (`static_group_membership_boosts.boosts`, `finance_act_general.boost`)
- Modify: `packages/common/src/common/es_client.py:548-559` (comment only — the code
  itself just reads `_BOOST_CONFIG`, no logic change needed there)
- Test: `packages/common/tests/test_es_client.py` (search for `_static_group_should_clauses`
  or `test_static_group` to find existing tests to update)

**Interfaces:**
- Consumes: nothing new
- Produces: `_static_group_should_clauses() -> list[dict]` — same signature, corrected
  values

### Background

Real source (`GlobalSearchResearch.cs:690-698, 722`, `GetGlobalSearchQuery`'s positive
OR-list) uses weights in the 15000-35000 range per subgroup, plus `FinanceActBoostquery` at
30000. This repo's own config file already records, in its own comment
(`repotaxmannapi_boost_config.json:46`), that these were "scaled by the same ~0.25 factor"
— so the real values are recoverable exactly by dividing each current value by 0.25 (i.e.
multiplying by 4), confirmed against the file's current contents (read 2026-09-06):

| subgroup_id | current (scaled) | real (÷0.25) |
|---|---|---|
| `111050000000017082` (CGST 2017) | 8750.0 | **35000.0** |
| `111050000000011411` (Companies Act 2013) | 6250.0 | **25000.0** |
| `111050000000017185` | 4000.0 | **16000.0** |
| `111050000000012771` | 3750.0 | **15000.0** |
| `111050000000017818` | 3750.0 | **15000.0** |
| `finance_act_general.boost` | 7500.0 | **30000.0** |

The user's explicit instruction is total copy-paste, not a re-tuned scale, so this must be
corrected to the real numbers above.

- [ ] **Step 1: Write the failing test asserting the real values**

```python
def test_static_group_should_clauses_use_real_unscaled_weights():
    """2026-09-06 correction: these were previously scaled ~0.25x from the real source's
    literal weights (GlobalSearchResearch.cs:690-698/722) to fit this repo's should-clause
    scale - the user's explicit instruction is byte-exact copy-paste, so this asserts the
    real, unscaled numbers (repotaxmannapi_boost_config.json's own comment already recorded
    the exact 0.25 scale factor used, making the real values recoverable: multiply each
    current value by 4)."""
    clauses = _static_group_should_clauses()
    boosts_by_subgroup = {
        c["term"]["groups.group.subgroup.id"]["value"]: c["term"]["groups.group.subgroup.id"]["boost"]
        for c in clauses if "term" in c
    }
    assert boosts_by_subgroup["111050000000017082"] == 35000  # CGST 2017
    assert boosts_by_subgroup["111050000000011411"] == 25000  # Companies Act 2013
    assert boosts_by_subgroup["111050000000017185"] == 16000
    assert boosts_by_subgroup["111050000000012771"] == 15000
    assert boosts_by_subgroup["111050000000017818"] == 15000
    finance_act_clause = next(c for c in clauses if "bool" in c)
    assert finance_act_clause["bool"]["boost"] == 30000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/common/tests/test_es_client.py -k static_group -v`
Expected: FAIL (current values are the scaled ones)

- [ ] **Step 3: Update `repotaxmannapi_boost_config.json`'s values**

Edit `packages/common/src/common/data/repotaxmannapi_boost_config.json` lines 45-61
directly:

```json
  "static_group_membership_boosts": {
    "_comment": "Unconditional per-document group-membership should-boosts ported from SearchTextElastic.cs::GetGlobalSearchQuery's positive OR-list (line 785). Real, unscaled weights (GlobalSearchResearch.cs:690-698/722) - 2026-09-06 correction, a prior version scaled these ~0.25x to fit this repo's should-clause scale, reverted per explicit user instruction to copy-paste real production logic exactly. [subgroup_id, boost] pairs.",
    "boosts": [
      ["111050000000017082", 35000.0],
      ["111050000000011411", 25000.0],
      ["111050000000017185", 16000.0],
      ["111050000000012771", 15000.0],
      ["111050000000017818", 15000.0]
    ]
  },

  "finance_act_general": {
    "_comment": "Finance Act (general) current-edition should-boost (FinanceActBoostquery, real weight 30000 - 2026-09-06 correction, un-scaled from a prior 7500). NOT a hard exclusion - see docs/pending-data-followups.md / es_client.py's 2026-09-03 correction note for why an earlier version wrongly excluded every non-current-year edition.",
    "subgroup_id": "111050000000010567",
    "current_year": "2025",
    "boost": 30000.0
  },
```

Then update the comment at `es_client.py:548-559` (which today explains the ~0.25 scaling
rationale) to instead read:

```python
# Group-membership should-boosts ported from SearchTextElastic.cs::GetGlobalSearchQuery's
# positive OR-list (line 785, CatIdsQuery && (...)) - real, unscaled weights
# (GlobalSearchResearch.cs:690-698/722), verbatim per the user's explicit copy-paste
# instruction (2026-09-06 correction - a prior version scaled these ~0.25x, which is now
# reverted). Fire unconditionally (per-document subgroup membership only, no query-side
# gating).
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/common/tests/test_es_client.py -k static_group -v`
Expected: PASS

- [ ] **Step 5: Run the whole common suite**

Run: `uv run pytest packages/common/tests -q`
Expected: only the one pre-existing unrelated failure

- [ ] **Step 6: Commit**

```bash
git add packages/common/src/common/data/repotaxmannapi_boost_config.json packages/common/src/common/es_client.py packages/common/tests/test_es_client.py
git commit -m "fix(repotaxmannapi): use real unscaled static group-membership boost weights"
```

---

## Task 2: Restructure should-clause assembly into per-field-tier AND-of-OR

**Files:**
- Modify: `packages/common/src/common/repotaxmannapi_query_builder.py` (whole file — changes
  the return shape of `build_should_clauses`, `_pipe_split_clauses`, `_tx_global_clauses`,
  `_citation_clauses`)
- Modify: `packages/common/src/common/es_client.py:914-1004`
  (`_build_repotaxmannapi_field_query` — how the per-field-tier results get assembled into
  the final `should`/`bool_query`)
- Test: `packages/common/tests/test_repotaxmannapi_query_builder.py` (every existing test
  changes shape — see Step 2)
- Test: `packages/common/tests/test_es_client.py` (assembly-level tests)

**Interfaces:**
- Produces: `build_should_clauses(tokens, is_global, is_excus, group_id="0") ->
  dict[str, list[dict]]` — a dict keyed by field-tier name (`"heading"`, `"subheading"`,
  `"searchboosttext"`, `"headnotes_text"`, `"fullcontent"`, `"searchheadingnumber"`,
  `"otherinfo.fullcitation.name"`), each value being the list of **per-token OR-groups**
  for that field (each OR-group itself a `list[dict]` of 1+ alternative `match_phrase`
  clauses to be OR'd, one OR-group per token that contributed to this field). Downstream
  (`es_client.py`) ANDs the OR-groups together per field, then ORs the resulting per-field
  bool queries together at the top level — this is the exact shape change needed to match
  the real source's `queryHeadingAnd &= (heading1 || heading2)` (repeated per token,
  AND-accumulated) pattern.
- Consumes: same `RepotaxmannapiToken` list as before; `TokenType`, `ProximityDefault` from
  `repotaxmannapi_tokenizer`.

### Background (read this before writing code)

**The real structural pattern, confirmed by directly reading every branch of
`SearchTextElastic.cs`'s `GetQuery` (lines 838-1206):**

Every branch (default/else at 1084-1107, TX-global at 1010-1042, CT at 1043-1066, PH at
1068-1082, and the pipe-split branch at 838-928) does the SAME thing per field, just with a
different-sized OR-group of alternate boost tiers:

```csharp
queryHeadingAnd &= (heading1 || heading2);   // TX-global: 2 alternatives, one AND-step
queryHeadingAnd &= new ...MatchPhrase(...);  // CT/PH/default: 1 alternative, one AND-step
```

i.e., **for each token in the query, that token's own clauses for a given field (whether 1
alternative or 3) are OR'd together into one "OR-group", and that OR-group is then ANDed
onto a running per-field-tier container that accumulates across ALL tokens in the query.**
The pipe-split branch (838-928) does the identical thing, just building the running
container's contribution across a token's pipe-separated alternatives via a temporary
`queryHeadingOr` variable first (since a pipe-token's own alternatives are ALSO OR'd
together, matching every other branch's "OR within a token"), then ANDing that whole thing
into `queryHeadingAnd` once, at line 1154, alongside every other (non-pipe) token's direct
contributions.

After the per-token loop finishes, each field-tier's fully-accumulated AND container becomes
ONE entry in the `queries` list (lines 1154-1190):

```csharp
queries.Add(queryHeadingAnd &= queryHeadingOr);
queries.Add(querySubheadingAnd &= querySubheadingOr);
queries.Add(querySearchBoostTextAnd &= querySearchBoostTextOr);
queries.Add(queryHeadnotesAnd &= queryHeadnotesOr);
queries.Add(queryCorrespondingCitation);
queries.Add(queryFullcontentAnd);  // or the must-not-wrapped variant - see Task 3
queries.Add(querysearchheadingnumberOr);
```

Then (`GetGlobalSearchQuery`, not audited in full here since it's the caller, but confirmed
by the FunctionScore audit) all of `queries`' entries get OR'd together as the final
`should`-equivalent at the top level.

**What this repo's current code does instead:** `build_should_clauses` returns one FLAT
`list[dict]` with every token's every alternative appended as an independently-optional
`should` clause. This means, for a 2-token query like `"income tax"`, a document with
"income" in its heading but not "tax" still gets full credit for the heading tier's boost
from that one clause — something the real source's `queryHeadingAnd &= (income_clause) &=
(tax_clause)` would never award (the heading tier's AND container is null/non-matching
unless BOTH tokens' heading clauses are satisfied).

**The fix:** change `build_should_clauses` (and the branch helper functions it dispatches
to) to return, instead of one flat list, a `dict[str, list[list[dict]]]` — field name ->
list of per-token OR-groups (each OR-group itself a list of 1+ alternative clause dicts).
`es_client.py` then does the AND-of-OR assembly: for each field, wrap each OR-group as
`{"bool": {"should": <alternatives>, "minimum_should_match": 1}}` (or just the single
clause directly if the OR-group has exactly 1 alternative — no need for a wrapping bool in
that case), then AND all of a field's OR-groups together via `{"bool": {"must": [...]}}`
(or the single group directly if there's only one). Finally, OR all field-tier results
together as the top-level `should` (this preserves current behavior for a single-token
query — a 1-token query's AND-of-1-OR-group per field degenerates to exactly what today's
flat list already produces for that case, so single-token-query tests should be unaffected
in outcome, only in the intermediate data shape).

- [ ] **Step 1: Read the current file fully before touching anything**

Read `packages/common/src/common/repotaxmannapi_query_builder.py` in full (441 lines) and
`packages/common/src/common/es_client.py:914-1004` before writing any code — the exact
current boost values, slop rules, and field names must be preserved exactly, only the
RETURN SHAPE changes.

- [ ] **Step 2: Rewrite `_pipe_split_clauses` to return per-alternative OR-groups grouped by field**

Replace the function's flat-`should`-returning body with one that groups by field, one
list-of-alternatives per field (the pipe-alternatives ARE the OR-group for this one field
for this ONE token, matching the real source's per-token OR-then-AND-across-tokens
pattern):

```python
def _pipe_split_clauses(token: RepotaxmannapiToken, group_id: str) -> dict[str, list[dict]]:
    """Pipe-separated ("|") OR-group branch, SearchTextElastic.cs:838-928. Returns ONE
    OR-group per field for this token - the pipe-alternatives (SearchTextElastic.cs:841's
    `qt.QueryText.Split('|')`) ARE the token's OR-group for each field, matching every
    other branch's "this token's own alternatives are OR'd, then ANDed onto the running
    per-field container across all tokens in the query" pattern (see
    build_should_clauses's docstring and this module's own "Background" note in the
    2026-09-06 parity plan). Field keys match _PHRASE_BOOSTS_STANDARD's names plus
    "searchheadingnumber" when the CirNot condition fires for any alternative."""
    groups: dict[str, list[dict]] = {
        "heading": [], "subheading": [], "searchboosttext": [], _HEADNOTES_TEXT_FIELD: [],
        "fullcontent": [],
    }
    minus_group: list[dict] = []
    searchheadingnumber_group: list[dict] = []
    for alt in token.query_text.split("|"):  # SearchTextElastic.cs:841
        query = alt  # normalization at :848-860 not ported here - see Task 5
        groups["heading"].append(
            {"match_phrase": {"heading": {
                "query": query, "boost": 155000, "slop": token.proximity, "analyzer": "snowball",
            }}}
        )  # SearchTextElastic.cs:877/900
        groups["subheading"].append(
            {"match_phrase": {"subheading": {
                "query": query, "boost": 80000, "slop": token.proximity, "analyzer": "snowball",
            }}}
        )  # SearchTextElastic.cs:878/901
        groups["searchboosttext"].append(
            {"match_phrase": {"searchboosttext": {
                "query": query, "boost": 70000, "slop": token.proximity, "analyzer": "snowball",
            }}}
        )  # SearchTextElastic.cs:879/902
        groups[_HEADNOTES_TEXT_FIELD].append(
            {"match_phrase": {_HEADNOTES_TEXT_FIELD: {
                "query": query, "boost": 65000, "slop": token.proximity, "analyzer": "snowball",
            }}}
        )  # SearchTextElastic.cs:884/907

        fc_slop = 10000 if token.proximity == ProximityDefault.DEFAULT_VALUE else token.proximity
        groups["fullcontent"].append(
            {"match_phrase": {"fullcontent": {
                "query": query, "boost": 1, "slop": fc_slop, "analyzer": "snowball",
            }}}
        )  # SearchTextElastic.cs:885-886/908-909

        if token.type == "T1" and "SECTION " in query.strip():
            minus_group.append(
                {"match_phrase": {"fullcontent": {
                    "query": f"SUB {query}", "boost": 1, "slop": token.proximity,
                    "analyzer": "snowball",
                }}}
            )  # SearchTextElastic.cs:890/913
        elif token.type == "T1" and group_id == _CIRNOT_GROUP_ID:
            searchheadingnumber_group.append(
                {"match_phrase": {"searchheadingnumber": {
                    "query": query, "boost": 85000, "slop": token.proximity,
                    "analyzer": "snowball",
                }}}
            )  # SearchTextElastic.cs:894/917

    result = {k: v for k, v in groups.items() if v}
    if minus_group:
        result["_fullcontent_minus"] = minus_group
    if searchheadingnumber_group:
        result["searchheadingnumber"] = searchheadingnumber_group
    return result
```

Note the new `"_fullcontent_minus"` key (leading underscore signals "not a real ES field
name, consumed specially by the assembly step" — see Step 5/Task 3) replacing the old
flat-list minus-clause entry.

- [ ] **Step 3: Rewrite `_tx_global_clauses` and `_citation_clauses` the same way**

Both currently return a flat `list[dict]` for ONE token; change them to return
`dict[str, list[dict]]` (field name -> that token's own OR-group of alternatives) using the
exact same boost/slop values already in the file (do not change any number, only the
return shape/grouping):

```python
def _tx_global_clauses(token: RepotaxmannapiToken) -> dict[str, list[dict]]:
    """... (keep the existing docstring, update only the "returns" description to say
    'one OR-group per field, keyed by field name' instead of 'a flat should-list')."""
    query = token.query_text
    groups: dict[str, list[dict]] = {}

    def _mp(field: str, boost: int, slop: int) -> dict:
        return {
            "match_phrase": {
                field: {"query": query, "boost": boost, "slop": slop, "analyzer": "snowball"}
            }
        }

    groups["heading"] = [_mp("heading", 155000, token.proximity - 4), _mp("heading", 90000, token.proximity)]
    groups["subheading"] = [_mp("subheading", 80000, token.proximity - 4), _mp("subheading", 75000, token.proximity)]
    groups["searchboosttext"] = [_mp("searchboosttext", 70000, token.proximity - 4), _mp("searchboosttext", 67000, token.proximity)]
    groups[_HEADNOTES_TEXT_FIELD] = [
        _mp(_HEADNOTES_TEXT_FIELD, 65000, token.proximity - 4),
        _mp(_HEADNOTES_TEXT_FIELD, 60000, token.proximity),
        _mp(_HEADNOTES_TEXT_FIELD, 50000, 100),
    ]
    if " " in query:
        groups["fullcontent"] = [_mp("fullcontent", 100, token.proximity), _mp("fullcontent", 1, 5000)]
    else:
        groups["fullcontent"] = [_mp("fullcontent", 1, 5000)]
    return groups
```

```python
def _citation_clauses(token: RepotaxmannapiToken) -> dict[str, list[dict]]:
    """... (keep existing docstring, update returns description same as above)."""
    query = token.query_text
    groups: dict[str, list[dict]] = {
        "heading": [{"match_phrase": {"heading": {
            "query": query, "boost": 155000, "slop": token.proximity, "analyzer": "snowball",
        }}}],
        "subheading": [{"match_phrase": {"subheading": {
            "query": query, "boost": 80000, "slop": token.proximity, "analyzer": "snowball",
        }}}],
        "searchboosttext": [{"match_phrase": {"searchboosttext": {
            "query": query, "boost": 70000, "slop": token.proximity, "analyzer": "snowball",
        }}}],
        _HEADNOTES_TEXT_FIELD: [{"match_phrase": {_HEADNOTES_TEXT_FIELD: {
            "query": query, "boost": 65000, "slop": token.proximity, "analyzer": "snowball",
        }}}],
        "otherinfo.fullcitation.name": [{"match_phrase": {"otherinfo.fullcitation.name": {
            "query": query, "boost": 155000, "slop": token.proximity, "analyzer": "snowball",
        }}}],
    }
    if token.type == TokenType.TEXT:
        fc_slop = 10000
    elif token.proximity == ProximityDefault.DEFAULT_VALUE:
        fc_slop = 10000
    else:
        fc_slop = token.proximity
    groups["fullcontent"] = [{"match_phrase": {"fullcontent": {
        "query": query, "boost": 1, "slop": fc_slop, "analyzer": "snowball",
    }}}]
    return groups
```

- [ ] **Step 4: Rewrite `build_should_clauses` to accumulate per-token OR-groups per field, across all tokens**

```python
def build_should_clauses(
    tokens: list[RepotaxmannapiToken], is_global: bool, is_excus: bool, group_id: str = "0",
) -> dict[str, list[list[dict]]]:
    """Build the per-field-tier AND-of-OR should-clause structure for a list of tokens,
    mirroring SearchTextElastic.cs's GetQuery per-token loop (838-1206): for EACH field
    (heading/subheading/searchboosttext/headnotes_text/fullcontent/searchheadingnumber/
    otherinfo.fullcitation.name), this returns the list of per-token OR-groups that must
    ALL be satisfied (AND) for that field-tier to contribute - each OR-group being one
    token's own alternative boost tiers (1-3 entries), which need only ONE to match.
    `es_client.py`'s assembly step ANDs these OR-groups together per field, then ORs the
    resulting per-field bool queries at the top level - see the 2026-09-06 parity plan's
    Task 2 for the full real-source citation this restructure is based on.

    A field with an empty OR-group list means no token contributed to it at all (mirrors
    the real source's untouched, always-true default QueryContainer - omit that field's
    tier entirely from the final query, contributing nothing, rather than emitting a
    vacuous match-everything clause).

    Field_suffix (".phrase_search" when is_excus) applies to every field this function
    itself builds directly (the non-pipe/non-TX/non-CT default fallthrough loop) - the
    three helper functions (_pipe_split_clauses, _tx_global_clauses, _citation_clauses)
    are dispatched to independent of is_excus per this module's existing convention (see
    each function's own docstring)."""
    field_suffix = ".phrase_search" if is_excus else ""
    result: dict[str, list[list[dict]]] = {}

    def _add_group(field: str, or_group: list[dict]) -> None:
        result.setdefault(field, []).append(or_group)

    for token in tokens:
        if "|" in token.query_text:
            per_field = _pipe_split_clauses(token, group_id)
            for field, or_group in per_field.items():
                _add_group(field, or_group)
            continue
        if token.type == TokenType.CITATION:
            per_field = _citation_clauses(token)
            for field, or_group in per_field.items():
                _add_group(field, or_group)
            continue
        if is_global and not is_excus and token.type == TokenType.TEXT:
            per_field = _tx_global_clauses(token)
            for field, or_group in per_field.items():
                _add_group(field, or_group)
            continue

        for field, boost in _PHRASE_BOOSTS_STANDARD.items():
            field_name = f"{field}{field_suffix}"
            if field == "fullcontent" and not is_excus:
                if token.type == TokenType.TEXT:
                    slop = 10000
                elif token.proximity == ProximityDefault.DEFAULT_VALUE:
                    slop = 10000
                else:
                    slop = token.proximity
            else:
                slop = token.proximity
            match_phrase: dict = {"query": token.query_text, "boost": boost, "slop": slop}
            if not is_excus and field in _SNOWBALL_ANALYZER_FIELDS:
                match_phrase["analyzer"] = "snowball"
            _add_group(field_name, [{"match_phrase": {field_name: match_phrase}}])

        if token.type == "T1" and "SECTION " in token.query_text:
            fullcontent_field = f"fullcontent{field_suffix}"
            minus_match_phrase: dict = {
                "query": f"SUB {token.query_text}", "boost": _SUB_EXCLUSION_BOOST,
                "slop": token.proximity,
            }
            if not is_excus:
                minus_match_phrase["analyzer"] = "snowball"
            _add_group("_fullcontent_minus", [{"match_phrase": {fullcontent_field: minus_match_phrase}}])

    return result
```

- [ ] **Step 5: Rewrite every existing test in `test_repotaxmannapi_query_builder.py` for the new return shape**

Read the current file fully first. Every test currently does something like:

```python
clauses = build_should_clauses([token], is_global=False, is_excus=False)
boosts_by_field = {
    list(c["match_phrase"].keys())[0]: list(c["match_phrase"].values())[0]["boost"]
    for c in clauses if "match_phrase" in c
}
```

Rewrite each to index into the new dict-of-list-of-lists shape instead. A helper at the
top of the test file makes this manageable:

```python
def _flatten_or_groups(result: dict[str, list[list[dict]]], field: str) -> list[dict]:
    """Test helper: flatten one field's list of OR-groups back into a single flat list of
    match_phrase clause dicts, for tests that only care about which clauses exist for a
    field, not the AND/OR grouping itself."""
    return [clause for or_group in result.get(field, []) for clause in or_group]
```

Then, e.g., `test_builds_phrase_boost_should_clauses_for_a_plain_text_token` becomes:

```python
def test_builds_phrase_boost_should_clauses_for_a_plain_text_token():
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
```

Apply the same mechanical transform to every other existing test in the file (the pipe-split
tests, the TX-global tests, the citation tests, the section-minus-clause tests — for the
minus-clause tests, read from `result["_fullcontent_minus"]` instead of filtering
`"fullcontent"` for a `"SUB "`-prefixed query). Do not skip any — every existing assertion
must still hold, just reached through the new shape.

- [ ] **Step 6: Run the query-builder tests to verify they compile and reveal any shape mismatches**

Run: `uv run pytest packages/common/tests/test_repotaxmannapi_query_builder.py -v`
Expected: initially FAIL or ERROR wherever a test wasn't yet updated for the new shape —
fix each one until all pass.

- [ ] **Step 7: Rewrite `_build_repotaxmannapi_field_query`'s assembly in `es_client.py`**

Read `es_client.py:914-1004` fully first. Replace the `should = build_should_clauses(...)`
+ flat-list assembly with per-field AND-of-OR assembly:

```python
def _and_of_or_groups(or_groups: list[list[dict]]) -> dict:
    """AND together a field's per-token OR-groups (each itself OR'd internally) -
    SearchTextElastic.cs's queryHeadingAnd &= (alt1 || alt2) pattern, repeated per token,
    now reconstructed from build_should_clauses's per-field-tier output. A single OR-group
    degenerates to that group directly (no wrapping needed) when there's only one token's
    worth of contribution to this field - matches today's single-token-query behavior
    exactly."""
    wrapped = [
        or_group[0] if len(or_group) == 1 else {"bool": {"should": or_group, "minimum_should_match": 1}}
        for or_group in or_groups
    ]
    return wrapped[0] if len(wrapped) == 1 else {"bool": {"must": wrapped}}


def _build_repotaxmannapi_field_query(query: str) -> dict:
    """... (keep the existing module-level docstring about FunctionScore always wrapping,
    add a note that the should-list is now assembled per-field via AND-of-OR, matching
    SearchTextElastic.cs's queryFieldAnd/queryFieldOr accumulation exactly - see the
    2026-09-06 parity plan's Task 2)."""
    tokens = tokenize(query)
    phrase_tokens = [t for t in tokens if t.type == TokenType.PHRASE_WORD]
    other_tokens = [t for t in tokens if t.type != TokenType.PHRASE_WORD]
    group_id = next((t.group_id for t in tokens if t.group_id != "0"), "0")

    per_field = build_should_clauses(other_tokens, is_global=True, is_excus=False, group_id=group_id)
    if phrase_tokens:
        phrase_per_field = build_should_clauses(phrase_tokens, is_global=True, is_excus=True, group_id=group_id)
        for field, or_groups in phrase_per_field.items():
            per_field.setdefault(field, []).extend(or_groups)

    minus_groups = per_field.pop("_fullcontent_minus", [])
    should = []
    for field, or_groups in per_field.items():
        field_query = _and_of_or_groups(or_groups)
        if field == "fullcontent" and minus_groups:
            # SearchTextElastic.cs:1169: queryFullcontentAnd && queryFullcontentOr &&
            # !queryFullcontentMinusOr - the SUB-clause is a NEGATION of this field-tier's
            # own contribution, not a positive boost. See Task 3.
            minus_query = _and_of_or_groups(minus_groups)
            field_query = {"bool": {"must": [field_query], "must_not": [minus_query]}}
        should.append(field_query)

    if not other_tokens and phrase_tokens:
        bool_query = {
            "bool": {
                "must": [{"bool": {"should": should, "minimum_should_match": 1}}],
                "should": _static_group_should_clauses(),
                "filter": [_edition_exclusion_filter(), *_additional_exclusion_filters()],
            },
        }
    else:
        should.extend(_static_group_should_clauses())
        bool_query = {
            "bool": {
                "should": should, "minimum_should_match": 1,
                "filter": [_edition_exclusion_filter(), *_additional_exclusion_filters()],
            },
        }
    return {
        "function_score": {
            "query": bool_query,
            "functions": build_function_score_functions(
                group_id,
                latest_finance_act_year=_BOOST_CONFIG["latest_finance_act_year_for_multiply_formula"]["value"],
            ),
            "score_mode": "multiply",
            "boost_mode": "multiply",
        },
    }
```

Note: Task 3 (below) replaces the placeholder `if field == "fullcontent" and minus_groups:`
block above with the final version — this step just needs the code to run and the tests
from Steps 8-9 to pass with the minus-clause STILL present as a positive should entry
temporarily is NOT acceptable (that's the Task-3 bug) — so implement the `must_not` version
directly here, don't stage it as a TODO. The block shown above already IS the Task 3 fix;
Task 3 exists mainly to add its own dedicated regression test, not because the code needs
a second pass.

- [ ] **Step 8: Update `test_es_client.py`'s two whole-query-phrase tests for the new nested shape**

The `must[0]["bool"]["should"]` in
`test_raw_search_repotaxmannapi_whole_query_phrase_still_wraps_function_score` and
`test_raw_search_repotaxmannapi_whole_query_phrase_requires_text_match` now contains
per-field AND-of-OR bool queries (or single clauses) instead of flat `match_phrase` dicts
directly. Update the assertions to look for `"bool"` wrapping where a field has multiple
tokens' worth of contribution, or a direct clause where it has only one - for the
single-quoted-phrase test cases these plans already use, there's exactly one PHRASE_WORD
token, so each field-tier still degenerates to exactly one direct `match_phrase` clause
(no AND wrapping needed) - the existing assertions (`"match_phrase" in c and
"heading.phrase_search" in c["match_phrase"]`) should keep working since the single-token
case produces the same shape as before. Run the tests first (Step 9) to confirm, and only
change what actually breaks.

- [ ] **Step 9: Run the whole common suite**

Run: `uv run pytest packages/common/tests -q`
Expected: only the one pre-existing unrelated failure. Fix any test file this restructure
touches that Step 8 didn't anticipate.

- [ ] **Step 10: Commit**

```bash
git add packages/common/src/common/repotaxmannapi_query_builder.py packages/common/src/common/es_client.py packages/common/tests/test_repotaxmannapi_query_builder.py packages/common/tests/test_es_client.py
git commit -m "fix(repotaxmannapi): restructure should-clauses into per-field AND-of-OR, matching real source"
```

---

## Task 3: Regression test for the SUB-exclusion clause now being a real negation

**Files:**
- Test: `packages/common/tests/test_es_client.py`

**Interfaces:**
- Consumes: `_build_repotaxmannapi_field_query` (already fixed by Task 2, Step 7)

### Background

Real source (`SearchTextElastic.cs:1169`): `queryFullcontentAnd && queryFullcontentOr &&
!queryFullcontentMinusOr` — the `"SUB " + query`-matching clause EXCLUDES documents, it does
not boost them. Task 2's Step 7 already implements the fix as part of the restructure (the
`must_not` wrapping). This task exists purely to add a dedicated, explicit regression test
so the fix can never silently regress back to a positive boost.

- [ ] **Step 1: Write the failing-if-regressed test**

```python
@pytest.mark.asyncio
async def test_raw_search_repotaxmannapi_section_minus_clause_is_an_exclusion_not_a_boost():
    """SearchTextElastic.cs:1169: `queryFullcontentAnd && queryFullcontentOr &&
    !queryFullcontentMinusOr` - the "SUB " + query fullcontent clause is a NEGATION
    (excludes documents whose fullcontent contains a sub-section back-reference to this
    section number), not a positive should-boost. A prior version of this code added it as
    a positive `should` clause with boost 1 - this is the regression guard."""
    client = FakeAsyncES(search_hits=[])

    await raw_search(client, "Section 92C", limit=20, boost=True, boost_source="repotaxmannapi")

    query = client.search_calls[0]
    bool_query = query["function_score"]["query"]["bool"]
    fullcontent_entries = [
        c for c in bool_query["should"]
        if "bool" in c and "must_not" in c.get("bool", {})
    ]
    assert len(fullcontent_entries) == 1, "expected exactly one fullcontent tier with a must_not exclusion"
    must_not = fullcontent_entries[0]["bool"]["must_not"]
    assert len(must_not) == 1
    minus_clause = must_not[0]
    assert "match_phrase" in minus_clause
    assert minus_clause["match_phrase"]["fullcontent"]["query"] == "SUB SECTION 92C"
    # confirm it is NOT also present as a positive should clause anywhere
    assert not any(
        "match_phrase" in c and c["match_phrase"].get("fullcontent", {}).get("query") == "SUB SECTION 92C"
        for c in bool_query["should"]
        if "bool" not in c
    )
```

- [ ] **Step 2: Run it**

Run: `uv run pytest packages/common/tests/test_es_client.py -k section_minus_clause_is_an_exclusion -v`
Expected: PASS (Task 2 already implemented the fix — this step only confirms it)

- [ ] **Step 3: Run the whole common suite**

Run: `uv run pytest packages/common/tests -q`
Expected: only the one pre-existing unrelated failure

- [ ] **Step 4: Commit**

```bash
git add packages/common/tests/test_es_client.py
git commit -m "test(repotaxmannapi): regression guard for SUB-clause being an exclusion, not a boost"
```

---

## Task 4: Add the secondary headnotestext OR-tier (boost 50000, slop 100)

**Files:**
- Modify: `packages/common/src/common/repotaxmannapi_query_builder.py`
  (`build_should_clauses`'s default-branch field loop)
- Test: `packages/common/tests/test_repotaxmannapi_query_builder.py`

**Interfaces:**
- Consumes/Produces: same `build_should_clauses` signature from Task 2

### Background

Real source (`SearchTextElastic.cs:1093-1097`, the plain "else" default branch):

```csharp
var Headnotes2 = new QueryContainer();
var Headnotes1 = ...MatchPhrase(m => m.Boost(65000).Field(f => f.headnotestext).Query(query).Analyzer("snowball").Slop(qt.QProximity));
if (qt.QType != "NZ")
     Headnotes2 = ...MatchPhrase(m => m.Boost(50000).Field(f => f.headnotestext).Query(query).Analyzer("snowball").Slop(100));
queryHeadnotesAnd &= (Headnotes1 || Headnotes2);
```

The primary 65000-boost tier (`_PHRASE_BOOSTS_STANDARD["headnotes_text"]`) is already
correctly ported. The secondary 50000-boost, fixed-slop-100 tier, OR'd with the primary one
for every token EXCEPT `NUM_ALPHA_ZONE` ("NZ") type, is currently missing entirely from the
default branch's field loop (it's already correctly present in `_tx_global_clauses` as
Headnotes3 — this task only adds it to the OTHER branch, the plain default one).

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/common/tests/test_repotaxmannapi_query_builder.py -k headnotestext_has_a_secondary -v`
Expected: FAIL (secondary tier absent today)

- [ ] **Step 3: Add the secondary tier to `build_should_clauses`'s default-branch loop**

In the `for field, boost in _PHRASE_BOOSTS_STANDARD.items():` loop (from Task 2's Step 4),
after appending the primary clause for `_HEADNOTES_TEXT_FIELD`, extend that same OR-group
with the secondary tier when the token isn't NUM_ALPHA_ZONE:

```python
        for field, boost in _PHRASE_BOOSTS_STANDARD.items():
            field_name = f"{field}{field_suffix}"
            if field == "fullcontent" and not is_excus:
                if token.type == TokenType.TEXT:
                    slop = 10000
                elif token.proximity == ProximityDefault.DEFAULT_VALUE:
                    slop = 10000
                else:
                    slop = token.proximity
            else:
                slop = token.proximity
            match_phrase: dict = {"query": token.query_text, "boost": boost, "slop": slop}
            if not is_excus and field in _SNOWBALL_ANALYZER_FIELDS:
                match_phrase["analyzer"] = "snowball"
            or_group = [{"match_phrase": {field_name: match_phrase}}]
            # Secondary headnotestext OR-tier, SearchTextElastic.cs:1093-1097 - boost
            # 50000, fixed slop 100, ORed with the primary tier for every token type
            # except NUM_ALPHA_ZONE ("NZ").
            if field == _HEADNOTES_TEXT_FIELD and not is_excus and token.type != TokenType.NUM_ALPHA_ZONE:
                secondary: dict = {"query": token.query_text, "boost": 50000, "slop": 100, "analyzer": "snowball"}
                or_group.append({"match_phrase": {field_name: secondary}})
            _add_group(field_name, or_group)
```

(Note: `TokenType.NUM_ALPHA_ZONE` already exists in `repotaxmannapi_tokenizer.py` per the
audit — confirm the exact attribute name matches, `TokenType.NUM_ALPHA_ZONE == "NZ"`, before
using it.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/common/tests/test_repotaxmannapi_query_builder.py -k headnotestext_has_a_secondary -v`
Expected: PASS

- [ ] **Step 5: Run the whole common suite**

Run: `uv run pytest packages/common/tests -q`
Expected: only the one pre-existing unrelated failure

- [ ] **Step 6: Commit**

```bash
git add packages/common/src/common/repotaxmannapi_query_builder.py packages/common/tests/test_repotaxmannapi_query_builder.py
git commit -m "feat(repotaxmannapi): add secondary headnotestext OR-tier (boost 50000, slop 100)"
```

---

## Task 5: `taxmann com`/`compcase` query-text normalization

**Files:**
- Modify: `packages/common/src/common/repotaxmannapi_query_builder.py`
- Test: `packages/common/tests/test_repotaxmannapi_query_builder.py`

### Background

Real source, both the pipe-split branch (`SearchTextElastic.cs:848-860`) and — per the
`_tx_global_clauses`/default-branch normalization note the module's own docstring already
flags as unported (the `query`/`querySearchboosttext` normalization, cs:935-947) — rewrites
query text before building any clause:

```csharp
if (x.ToLower().IndexOf("taxmann com") >= 0)
    query = x.ToLower().Replace("taxmann com", "taxmann.com");
else if (qt.QueryText.ToLower().IndexOf("compcase") >= 0)
    query = qt.QueryText.ToLower().Replace("compcase", "comp case");
else
    query = x;
querySearchboosttext = (x.ToLower().IndexOf("taxmann.com") >= 0) ? x.ToLower().Replace("taxmann.com", "taxmann com") : x;
```

i.e.: the `heading`/`subheading`/`headnotestext`/`fullcontent` fields' query text gets
`"taxmann com"` rewritten to `"taxmann.com"` (or `"compcase"` to `"comp case"`), while the
`searchboosttext` field's query text is rewritten the OPPOSITE direction (`"taxmann.com"` to
`"taxmann com"`) — two independently-computed strings, `query` and `querySearchboosttext`,
used for different fields in the same clause set.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/common/tests/test_repotaxmannapi_query_builder.py -k "taxmann or compcase" -v`
Expected: FAIL

- [ ] **Step 3: Add a shared normalization helper and wire it into every branch**

Add near the top of `repotaxmannapi_query_builder.py` (after the imports):

```python
def _normalize_query_text(raw: str) -> tuple[str, str]:
    """SearchTextElastic.cs:848-860 (pipe-split branch) / :935-947 (non-pipe branches,
    identical logic on qt.QueryText instead of a pipe-alternative). Returns
    (query, query_searchboosttext) - two independently-normalized strings: `query` feeds
    heading/subheading/headnotestext/fullcontent, `query_searchboosttext` feeds
    searchboosttext only (a separate, oppositely-directed rewrite)."""
    lowered = raw.lower()
    if "taxmann com" in lowered:
        query = lowered.replace("taxmann com", "taxmann.com")
    elif "compcase" in lowered:
        query = lowered.replace("compcase", "comp case")
    else:
        query = raw
    query_searchboosttext = raw.lower().replace("taxmann.com", "taxmann com") if "taxmann.com" in lowered else raw
    return query, query_searchboosttext
```

Then, in `build_should_clauses`'s default-branch loop, compute `query, query_sbt =
_normalize_query_text(token.query_text)` once per token (before the `for field, boost in
...` loop) and use `query_sbt` specifically for the `searchboosttext` field's clause
(`field == "searchboosttext"`), `query` for every other field (including the minus-clause
and the T1 in-loop checks, which should test against the ORIGINAL `token.query_text` for
the `"SECTION "` prefix check — the real source's minus-clause condition at cs:887/910
checks `query.Trim()`, i.e. the ALREADY-normalized `query`, not the raw token text, so use
the normalized `query` there too). Apply the identical `_normalize_query_text` call inside
`_pipe_split_clauses` (per-alternative, replacing the current `query = alt` line) and
inside `_tx_global_clauses`/`_citation_clauses` (per-token, replacing the current `query =
token.query_text` line), using `query_sbt` for each function's own `searchboosttext` tier.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/common/tests/test_repotaxmannapi_query_builder.py -k "taxmann or compcase" -v`
Expected: PASS

- [ ] **Step 5: Run the whole common suite**

Run: `uv run pytest packages/common/tests -q`
Expected: only the one pre-existing unrelated failure — pay special attention to any
EXISTING test whose expected query text assumed no normalization (e.g. a test using literal
`"taxmann.com"` or `"compcase"` in a query string elsewhere in the file) and update it if
it now legitimately normalizes.

- [ ] **Step 6: Commit**

```bash
git add packages/common/src/common/repotaxmannapi_query_builder.py packages/common/tests/test_repotaxmannapi_query_builder.py
git commit -m "feat(repotaxmannapi): port taxmann.com/compcase query-text normalization"
```

---

## Task 6: `GroupFilterquery` boost-1000 should-clause + fallback act-url boost

**Files:**
- Modify: `packages/common/src/common/es_client.py` (`_build_repotaxmannapi_field_query`,
  and a new small helper near `_static_group_should_clauses`)
- Test: `packages/common/tests/test_es_client.py`

### Background

Real source (`GlobalSearchResearch.cs:751-758`, inside `GetGlobalSearchQuery`):

```csharp
if (search.IsGlobal && searchProcess.iGroupID != "" && searchProcess.iGroupID != "0")
    globalQuery |= new QueryContainerDescriptor<ResearchIndexDocument>().MatchPhrase(m => m.Boost(1000).Field(f => f.groups.group.id).Query(searchProcess.iGroupID));
...
if (nullcount == queries.Count)
    globalQuery |= new QueryContainerDescriptor<ResearchIndexDocument>().Match(m => m.Boost(1000).Field(f => f.groups.group.url).Query(Constants_GroupUrl.Act));
```

Two independent should-boosts: (1) whenever the query resolved a non-`"0"` `iGroupID`
(the same `group_id` this repo already threads through `build_function_score_functions`),
add a `match_phrase` on `groups.group.id` == that group id, boost 1000; (2) when EVERY
per-token query ended up null (i.e., the tokenizer produced no usable tokens at all — an
edge case, but a real one, e.g. an all-stopword query), fall back to boosting
`groups.group.url == "act"` by 1000 via a plain `match` (not `match_phrase`).

`Constants_GroupUrl.Act` — grep `BL/Constants.cs` for its exact string value before using
it (expected `"act"`, confirm).

- [ ] **Step 1: Confirm `Constants_GroupUrl.Act`'s value**

```bash
grep -n "GroupUrl" "C:\A_VANSH\ALL_NEEDS\CODING\Taxmann\txmn-data-retrieval\misc\repotaxmannapi\TaxmannAPI\BL\Constants.cs"
```

- [ ] **Step 2: Write the failing tests**

```python
@pytest.mark.asyncio
async def test_raw_search_repotaxmannapi_adds_group_id_boost_should_clause_when_resolved():
    """GlobalSearchResearch.cs:751-754: search.IsGlobal && iGroupID not in ("", "0") ->
    match_phrase(groups.group.id == iGroupID, boost 1000)."""
    client = FakeAsyncES(search_hits=[])
    await raw_search(client, "Section 52 of Companies Act", limit=20, boost=True, boost_source="repotaxmannapi")
    query = client.search_calls[0]
    bool_query = query["function_score"]["query"]["bool"]
    group_id_clauses = [
        c for c in bool_query["should"]
        if "match_phrase" in c and "groups.group.id" in c["match_phrase"]
        and c["match_phrase"]["groups.group.id"].get("boost") == 1000
    ]
    assert len(group_id_clauses) == 1


@pytest.mark.asyncio
async def test_raw_search_repotaxmannapi_falls_back_to_act_url_boost_when_no_tokens_produced():
    """GlobalSearchResearch.cs:755-758: when every per-token query is null (nullcount ==
    queries.Count), fall back to match(groups.group.url == "act", boost 1000). A query
    that tokenizes to nothing usable (e.g. all stop words) triggers this."""
    client = FakeAsyncES(search_hits=[])
    await raw_search(client, "the of and", limit=20, boost=True, boost_source="repotaxmannapi")  # all stop words - adjust to a real all-stopword example if this doesn't tokenize to zero tokens
    query = client.search_calls[0]
    bool_query = query["function_score"]["query"]["bool"]
    fallback_clauses = [
        c for c in bool_query["should"]
        if "match" in c and "groups.group.url" in c["match"]
        and c["match"]["groups.group.url"].get("boost") == 1000
    ]
    assert len(fallback_clauses) == 1
```

(If `"the of and"` doesn't actually tokenize to zero real tokens in this port, find one
that does by checking `repotaxmannapi_token_dictionary`'s stop-word entries — grep the JSON
dictionary for `"element_type": "99"` entries and pick 2-3 of those words.)

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest packages/common/tests/test_es_client.py -k "group_id_boost or falls_back_to_act_url" -v`
Expected: FAIL

- [ ] **Step 4: Add both clauses in `_build_repotaxmannapi_field_query`**

After the `should.extend(_static_group_should_clauses())` line (in the `else` branch — the
general, non-whole-phrase case) and its equivalent spot for the whole-phrase branch (the
`_static_group_should_clauses()` call already present there too), add:

```python
    # GroupFilterquery, GlobalSearchResearch.cs:751-754: whenever a non-"0" group_id was
    # resolved (this is repotaxmannapi's own global-search path, search.IsGlobal is always
    # true here), boost documents whose groups.group.id matches it.
    if group_id != "0":
        should.append({"match_phrase": {"groups.group.id": {"query": group_id, "boost": 1000}}})

    # Fallback act-url boost, GlobalSearchResearch.cs:755-758: when the tokenizer produced
    # no usable tokens at all (every per-token query was null - `nullcount ==
    # queries.Count`), boost groups.group.url == "act" instead of leaving the query
    # entirely boost-less.
    if not tokens:
        should.append({"match": {"groups.group.url": {"query": "act", "boost": 1000}}})
```

Place this right before each `bool_query = {...}` construction (both the whole-phrase
branch and the general branch need it, since both share the same `should` list variable at
that point — check exactly where `should` is finalized in the current Task-2-restructured
code and insert once, before the branch that decides which `bool_query` shape to build, so
both paths get it).

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest packages/common/tests/test_es_client.py -k "group_id_boost or falls_back_to_act_url" -v`
Expected: PASS

- [ ] **Step 6: Run the whole common suite**

Run: `uv run pytest packages/common/tests -q`
Expected: only the one pre-existing unrelated failure

- [ ] **Step 7: Commit**

```bash
git add packages/common/src/common/es_client.py packages/common/tests/test_es_client.py
git commit -m "feat(repotaxmannapi): add GroupFilterquery boost-1000 clause + fallback act-url boost"
```

---

## Task 7: `MT`/`DM`/`DT` date-token branches

**Files:**
- Modify: `packages/common/src/common/repotaxmannapi_query_builder.py`
- Test: `packages/common/tests/test_repotaxmannapi_query_builder.py`

### Background

Confirmed by the tokenizer audit: `TokenType.MONTH_FMT` ("MT"), `TokenType.DATE_MONTH_FMT`
("DM"), `TokenType.DATE_FMT` ("DT") tokens are ALREADY correctly produced by
`repotaxmannapi_tokenizer.py` (fully ported) — this task is PURELY about `GetQuery`
consuming them, which currently has no branch for any of the three at all.

Real source (`SearchTextElastic.cs:948-1009`, read in full and verbatim-transcribed below
— every value here is directly confirmed against the live file, not reconstructed):

```csharp
if (qt.QType == "MT")
{
    var squeryMT = qt.QueryText.Split(' ');
    Month = Convert.ToString(TaxmannQueryAnalizer.MonthList()[squeryMT[0].ToUpper()]);  // numeric month string, e.g. "1".."12"
    Year = squeryMT.Length > 0 ? squeryMT[1] : "";
    queryDateField &= Wildcard(Boost(200000), documentdate, Value(Year + Month + "*"));  // e.g. "202401*"
    queryHeadingAnd &= MatchPhrase(Boost(155000), heading, Query(Month + " " + Year), Analyzer("snowball"), Slop(4));
    querySubheadingAnd &= MatchPhrase(Boost(80000), subheading, Query(Month + " " + Year), Slop(4));
    querySearchBoostTextAnd &= MatchPhrase(Boost(70000), searchboosttext, Query(Month + " " + Year), Slop(4));
    queryHeadnotesAnd &= MatchPhrase(Boost(65000), headnotestext, Query(Month + " " + Year), Slop(4));
    if (!search.isheadnoteToggle)
        queryFullcontentAnd &= MatchPhrase(Boost(1), fullcontent, Query(Month + " " + Year), Slop(4));
    isMT = true;
}
else if (qt.QType == "DM")
{
    var squeryDM = qt.QueryText.Split(' ');
    string Day = squeryDM.Length > 0 ? squeryDM[0] : "";
    string month = squeryDM.Length > 0 ? squeryDM[1] : "";
    Month = Convert.ToString(TaxmannQueryAnalizer.MonthList()[month.ToUpper()]);  // numeric month string
    queryDateField &= Wildcard(Boost(200000), documentdate, Value("*" + Month + Day));  // e.g. "*0115"
    queryHeadingAnd &= MatchPhrase(Boost(155000), heading, Query(Day + " " + Month), Slop(4));
    querySubheadingAnd &= MatchPhrase(Boost(80000), subheading, Query(Day + " " + Month), Slop(4));
    querySearchBoostTextAnd &= MatchPhrase(Boost(70000), searchboosttext, Query(Day + " " + Month), Slop(4));
    queryHeadnotesAnd &= MatchPhrase(Boost(65000), headnotestext, Query(Day + " " + Month), Slop(4));
    if (!search.isheadnoteToggle)
        queryFullcontentAnd &= MatchPhrase(Boost(1), fullcontent, Query(Day + " " + Month), Slop(4));
    isMT = true;  // NOTE: real source sets isMT=true here too, NOT a separate isDM flag - both MT and DM share the isMT downstream flag
}
else if (qt.QType == "DT")
{
    var squeryDMY = qt.QueryText.Split(' ');  // squeryDMY[0]=day, [1]=month(numeric), [2]=year - confirmed by the reversed concatenation below
    string formatteddocumentdate = squeryDMY[2] + squeryDMY[1] + squeryDMY[0];  // yyyy + MM + dd, e.g. "20240401"
    queryDateField &= Wildcard(Boost(200000), documentdate, Value(formatteddocumentdate));  // NO trailing "*" - exact-value Wildcard, not a real wildcard pattern
    querysearchhn &= MatchPhrase(Boost(10), headnotestext, Query(query), Analyzer("snowball"), Slop(qt.QProximity));
    querysearchfcontent &= MatchPhrase(Boost(1), fullcontent, Query(query), Slop(qt.QProximity));
    queryHeadingAnd &= MatchPhrase(Boost(155000), heading, Query(query), Slop(qt.QProximity));
    querySubheadingAnd &= MatchPhrase(Boost(80000), subheading, Query(query), Slop(qt.QProximity));
    querySearchBoostTextAnd &= MatchPhrase(Boost(70000), searchboosttext, Query(query), Slop(qt.QProximity));
    queryHeadnotesAnd &= MatchPhrase(Boost(65000), headnotestext, Query(query), Slop(qt.QProximity));
    if (!search.isheadnoteToggle)
    {
        queryFullcontentAnd &= MatchPhrase(Boost(1), fullcontent, Query(query), Slop(qt.QProximity));
        queryFullcontentAnd |= MatchPhrase(Boost(1), fullcontent, Query(searchProcess.SearchTextOrg), Slop(qt.QProximity));  // OR, not AND - a second fullcontent alternative using the whole original search text
    }
    isDT = true;
}
```

(`IsTopStory`/`IsTldSearch`-gated `topstoryheading`/`tldheading` tiers also present in all
three branches, same boosts/slop-4 pattern — out of scope, skip per this plan's UI-toggle
exclusion.)

The `documentdate` wildcard clause (boost 200000) is a NEW field-tier this repo's should-
list assembly doesn't have a slot for yet — treat it as its own field key
(`"documentdate"`) in the per-field-tier dict Task 2 established, so it flows through
`_and_of_or_groups`/the top-level `should.append(...)` loop identically to every other
field, no special-casing needed in `es_client.py`. Its ES query shape is
`{"wildcard": {"documentdate": {"value": ..., "boost": 200000}}}` (an ES `wildcard` query,
not `match_phrase` — no `analyzer`/`slop` apply to a wildcard query at all).

**Important divergence from a naive reading:** MT and DM both set the SAME `isMT` flag in
the real source (there is no separate `isDM`) — irrelevant to this port's per-token
function shape (that flag only affects the real source's later `queries.Add(queryDateField)`
step, which this port's field-key-based assembly in `es_client.py` doesn't need — every
non-empty field key, including `"documentdate"`, already gets OR'd into the top-level
`should` regardless of any `isMT`/`isDT` flag). **DT's wildcard has no trailing `"*"`** —
it's an exact-value match via the `wildcard` query type (unusual, but verbatim what the
real source does — do not add a `"*"` suffix).

`Year`/`Month`/`Day` for MT/DM are parsed by SPLITTING `qt.QueryText` on `' '` — NOT read
from `token.query_date`. This means the Python port needs `token.query_text` to already be
space-separated `"<MonthName> <Year>"` (MT) or `"<Day> <MonthName>"` (DM) — check
`repotaxmannapi_tokenizer.py`'s `_create_token` (already read during the tokenizer audit):
for `MONTH_FMT` it formats `query_text = f"{month_names()[query_date.month]} {query_date.year}"`
(month NAME + year — matches MT's expected `squeryMT[0]`=month-name, `squeryMT[1]`=year
shape) and for `DATE_MONTH_FMT` it formats `query_text = f"{query_date.day:02d}
{month_names()[query_date.month]}"` (day + month NAME — matches DM's expected
`squeryDM[0]`=day, `squeryDM[1]`=month-name shape). So: split `token.query_text` on `" "`
exactly like the real source does, look up the month NAME (second/first element
respectively) via `month_names()`'s own reverse mapping (or just use
`token.query_date.month`/`.year`/`.day` directly, which is simpler and equivalent since
`token.query_date` is always populated for these types per `_create_token` — prefer this
over re-parsing `query_text`, it's the same information without a fragile split/lookup
round-trip; the real source only re-derives from text because C# reformats through a
month-name dictionary, Python already has the structured `date` object). For DT,
`squeryDMY[0]/[1]/[2]` similarly correspond to `token.query_date.day`/`.month`/`.year`
zero-padded to 2/2/4 digits respectively (confirm `token.query_date` is populated for
`DATE_FMT` too — `_create_token`'s `DATE_FMT` branch already sets `query_date =
buffer_obj`, confirmed during the tokenizer audit).

- [ ] **Step 1: Format helpers, using values already confirmed against the real source**

`TaxmannQueryAnalizer.MonthList()` (cs:13-14) maps month name → an already zero-padded
2-digit string (`{"JAN":"01", "JANUARY":"01", ..., "DEC":"12", "DECEMBER":"12"}`) — so
`Month` in the real MT/DM branches is always exactly 2 characters, e.g. `"01"`. Use
`f"{token.query_date.month:02d}"` in Python (identical result, no dictionary round-trip
needed since Python already has the structured `date`). `Year` is used as-is from
`squeryMT[1]` (a plain string, whatever width the query text had) — use
`str(token.query_date.year)` (4 digits for any real date). `Day` similarly ->
`f"{token.query_date.day:02d}"`. Write these three one-line helpers (or inline
f-strings) at the top of `_date_field_clauses`:

```python
def _date_field_clauses(token: RepotaxmannapiToken) -> dict[str, list[dict]]:
    """MT/DM/DT branches, SearchTextElastic.cs:948-1009 (verbatim-transcribed values - see
    this plan's Task 7 Background). Month/Day/Year derived directly from
    `token.query_date` (always populated for these three token types per
    repotaxmannapi_tokenizer.py's _create_token) rather than re-parsing token.query_text,
    since that's the same information the real source's own text-split-then-dictionary-
    lookup produces, without the fragile round-trip."""
    assert token.query_date is not None
    d = token.query_date
    month2 = f"{d.month:02d}"  # MonthList()'s own values are already 2-digit, e.g. "01"
    day2 = f"{d.day:02d}"
    year = str(d.year)

    if token.type == TokenType.MONTH_FMT:  # MT
        wildcard_value = f"{year}{month2}*"
        field_query = f"{month2} {year}"
        slop = 4
    elif token.type == TokenType.DATE_MONTH_FMT:  # DM
        wildcard_value = f"*{month2}{day2}"
        field_query = f"{day2} {month2}"
        slop = 4
    else:  # DATE_FMT / DT
        wildcard_value = f"{year}{month2}{day2}"  # NO trailing "*" - cs:993, exact value
        field_query = token.query_text
        slop = token.proximity

    groups: dict[str, list[dict]] = {
        "documentdate": [{"wildcard": {"documentdate": {"value": wildcard_value, "boost": 200000}}}],
        "heading": [{"match_phrase": {"heading": {
            "query": field_query, "boost": 155000, "slop": slop, "analyzer": "snowball",
        }}}],
        "subheading": [{"match_phrase": {"subheading": {
            "query": field_query, "boost": 80000, "slop": slop, "analyzer": "snowball",
        }}}],
        "searchboosttext": [{"match_phrase": {"searchboosttext": {
            "query": field_query, "boost": 70000, "slop": slop, "analyzer": "snowball",
        }}}],
        _HEADNOTES_TEXT_FIELD: [{"match_phrase": {_HEADNOTES_TEXT_FIELD: {
            "query": field_query, "boost": 65000, "slop": slop, "analyzer": "snowball",
        }}}],
        "fullcontent": [{"match_phrase": {"fullcontent": {
            "query": field_query, "boost": 1, "slop": slop, "analyzer": "snowball",
        }}}],
    }
    # DT's second fullcontent OR-alternative (cs:1007, `queryFullcontentAnd |=
    # MatchPhrase(..., Query(searchProcess.SearchTextOrg), ...)`) needs the WHOLE
    # original, unsplit search-bar text - a value this per-token function has no access
    # to (same documented-gap pattern as elsewhere in this module for whole-query-level
    # values). Disclosed, not silently dropped: only DT's single-alternative fullcontent
    # tier above is emitted; the SearchTextOrg-based second alternative is NOT.
    return groups
```

- [ ] **Step 2: Write the failing tests**

```python
def test_month_format_token_builds_documentdate_wildcard_and_slop_4_field_tiers():
    # SearchTextElastic.cs:948-957 (MT branch) - exact values confirmed in Step 1.
    token = RepotaxmannapiToken(
        query_text="January 2024", org_text="Jan 2024",
        type="MT", or_in=False, proximity=1, query_date=date(2024, 1, 1),
    )
    result = build_should_clauses([token], is_global=True, is_excus=False)
    assert result["documentdate"][0][0]["wildcard"]["documentdate"]["boost"] == 200000
    heading_clause = result["heading"][0][0]
    assert heading_clause["match_phrase"]["heading"]["boost"] == 155000
    assert heading_clause["match_phrase"]["heading"]["slop"] == 4


def test_month_format_token_documentdate_wildcard_value_is_year_then_month():
    token = RepotaxmannapiToken(
        query_text="January 2024", org_text="Jan 2024",
        type="MT", or_in=False, proximity=1, query_date=date(2024, 1, 1),
    )
    result = build_should_clauses([token], is_global=True, is_excus=False)
    assert result["documentdate"][0][0]["wildcard"]["documentdate"]["value"] == "202401*"
    assert result["heading"][0][0]["match_phrase"]["heading"]["query"] == "01 2024"  # numeric month + " " + year, NOT the month name


def test_day_month_format_token_documentdate_wildcard_value_is_star_month_day():
    token = RepotaxmannapiToken(
        query_text="15 January", org_text="15 Jan",
        type="DM", or_in=False, proximity=1, query_date=date(2024, 1, 15),
    )
    result = build_should_clauses([token], is_global=True, is_excus=False)
    assert result["documentdate"][0][0]["wildcard"]["documentdate"]["value"] == "*0115"
    assert result["heading"][0][0]["match_phrase"]["heading"]["query"] == "15 01"  # day + " " + numeric month


def test_full_date_token_builds_documentdate_wildcard_with_no_trailing_star_and_qproximity_slop():
    # SearchTextElastic.cs:988-1009 (DT branch). NOTE the real source's Wildcard value has
    # NO trailing "*" for DT (unlike MT/DM) - an exact-value match via the wildcard query
    # type, confirmed by direct read of cs:993.
    token = RepotaxmannapiToken(
        query_text="01 04 2024", org_text="1-4-2024",
        type="DT", or_in=False, proximity=1, query_date=date(2024, 4, 1),
    )
    result = build_should_clauses([token], is_global=True, is_excus=False)
    wildcard_value = result["documentdate"][0][0]["wildcard"]["documentdate"]["value"]
    assert wildcard_value == "20240401"  # yyyy + MM + dd, no "*" suffix
    heading_clause = result["heading"][0][0]
    assert heading_clause["match_phrase"]["heading"]["slop"] == 1  # qt.QProximity, not fixed 4
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest packages/common/tests/test_repotaxmannapi_query_builder.py -k "month_format or day_month_format or full_date" -v`
Expected: FAIL (`"MT"`/`"DM"`/`"DT"` not dispatched to anything today, `result["documentdate"]`
doesn't exist)

- [ ] **Step 4: Add `_date_field_clauses` and dispatch to it from `build_should_clauses`**

Implement `_date_field_clauses(token: RepotaxmannapiToken) -> dict[str, list[dict]]`
mirroring the exact real-source values confirmed in Step 1, handling all three of
`TokenType.MONTH_FMT`, `TokenType.DATE_MONTH_FMT`, `TokenType.DATE_FMT` internally (branch
on `token.type`), returning field keys `"documentdate"`, `"heading"`, `"subheading"`,
`"searchboosttext"`, `_HEADNOTES_TEXT_FIELD`, `"fullcontent"` (and, for DT specifically, do
NOT introduce the `querysearchhn`/`querysearchfcontent` side-containers or the
`SearchTextOrg`-based second fullcontent OR-alternative — those depend on
`searchProcess.SearchTextOrg`, a whole-query-level value no per-token function here has
access to; document this as a disclosed gap in the function's docstring, matching this
module's existing convention for other whole-query-level gaps). Wire it into
`build_should_clauses`'s dispatch chain (before the pipe-split check, since a date token
could theoretically be pipe-joined too, though unlikely — check `"|" in token.query_text`
first regardless, matching every other branch's precedence):

```python
        if token.type in (TokenType.MONTH_FMT, TokenType.DATE_MONTH_FMT, TokenType.DATE_FMT):
            per_field = _date_field_clauses(token)
            for field, or_group in per_field.items():
                _add_group(field, or_group)
            continue
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest packages/common/tests/test_repotaxmannapi_query_builder.py -k "month_format or full_date" -v`
Expected: PASS

- [ ] **Step 6: Run the whole common suite**

Run: `uv run pytest packages/common/tests -q`
Expected: only the one pre-existing unrelated failure

- [ ] **Step 7: Commit**

```bash
git add packages/common/src/common/repotaxmannapi_query_builder.py packages/common/tests/test_repotaxmannapi_query_builder.py
git commit -m "feat(repotaxmannapi): port MT/DM/DT date-token query branches"
```

---

## Task 8: `ARTICLE` → `EXPERTSOPINION` token remap under global search

**Files:**
- Modify: `packages/common/src/common/repotaxmannapi_tokenizer.py`
- Modify: `packages/common/src/common/es_client.py` (the `tokenize(query)` call site in
  `_build_repotaxmannapi_field_query` needs to pass `is_global=True`)
- Test: `packages/common/tests/test_repotaxmannapi_tokenizer.py`

**Interfaces:**
- Produces: `tokenize(query: str, is_global: bool = False) -> list[RepotaxmannapiToken]` —
  new optional parameter, default `False` (preserves every existing caller's behavior
  unchanged)

### Background

Real source, directly read and verbatim-transcribed (`TaxmannQueryAnalizer.cs:1553-1554,
1594-1698, 1968-1984`) — **this plan's earlier draft of this task guessed a token-type
rewrite; that guess was wrong, corrected below from a direct read**:

`IsArticle`/`IsCountry` are two `bool` locals inside `ProcessorQuery` (initialized `false`
at 1553-1554), set at exactly 4 places:

```csharp
case ElementType.KeyWordOnly:                              // cs:1594-1601
    ...
    if (RText.ToUpper().StartsWith("ARTICLE")) IsArticle = true;
    break;
case ElementType.KeyWord:                                  // cs:1602-1625
    if (ProcessKeyWord(RText, ref sResult)) { ... }         // success path: no IsArticle set
    else {
        if (RText.ToUpper() != "AS") {
            ...
            if (RText.ToUpper().StartsWith("ARTICLE")) IsArticle = true;   // cs:1621, failure-path only
        }
    }
    break;
case ElementType.KeyWordType2:                              // cs:1626-1647
    if (ProcessKeyWordType2(...)) { ... }                    // success path: no IsArticle set
    else {
        ...
        if (RText.ToUpper().StartsWith("ARTICLE")) IsArticle = true;   // cs:1645, failure-path only
    }
    break;
...
case ElementType.Country:                                    // cs:~1690-1698
    ...
    IsCountry = true;                                         // unconditional in this branch
    break;
```

Then, at the very end of `ProcessorQuery`, AFTER the main scanning loop but BEFORE the
phrase-word tail-append (cs:1968-1984, immediately preceding the `PhraseElements` loop at
1985-1993):

```csharp
if (!string.IsNullOrEmpty(isGlobalSearch) && isGlobalSearch == "yes")
{
    if (IsArticle && !IsCountry)
    {
        for (iI = 0; iI <= QueryTokens.Count - 1; iI++)
        {
            TempToken = (Token)QueryTokens[iI];
            if (TempToken.QueryText.ToUpper().StartsWith("ARTICLE"))
            {
                TempToken.QueryText = GetKeySearchText("EXPERTSOPINION");
                TempToken.Proximity = GetKeyProximity("EXPERTSOPINION");
                QueryTokens[iI] = TempToken;
                ReSetPrimaryTag("EXPERTSOPINION");
            }
        }
    }
}
```

**Confirmed, corrected understanding — this rewrites `QueryText` and `Proximity` only,
NEVER the token's `Type`.** `ReSetPrimaryTag(KeyText)` (cs:318-344) is `SetPrimaryTag`
without its `iTagNo=="0"` guard — it unconditionally overwrites the query-level
`iTagNo`/`iProxiWords`/`iBoostFactor`/`iGroupID` from the `"EXPERTSOPINION"` dictionary
entry. This repo's port already stores `group_id` PER-TOKEN (a deliberate, documented
deviation from the real per-query-only `iGroupID` — see this module's own docstring), so
the most faithful available equivalent is to also overwrite the matched token's
`group_id` field to EXPERTSOPINION's own `group_id` when this remap fires for it.

The `"EXPERTSOPINION"` dictionary entry (confirmed present in
`packages/common/src/common/data/repotaxmannapi_token_dictionary.json`):
`{"element_type": "77", "tag_no": "7", "proximity": 2, "boost_factor": 100000, "group_id":
"111050000000000051", "search_text": "EXPERTS OPINION"}` — so `GetKeySearchText` yields
`"EXPERTS OPINION"` and `GetKeyProximity` yields `2`.

- [ ] **Step 1: Write the failing test**

```python
def test_article_token_query_text_and_proximity_remapped_to_expertsopinion_under_global_search():
    # TaxmannQueryAnalizer.cs:1968-1984, confirmed by direct read: rewrites QueryText to
    # "EXPERTS OPINION" and Proximity to 2 (the EXPERTSOPINION dictionary entry's own
    # values) - does NOT change the token's Type. Only fires when tokenize() is called
    # with is_global=True AND is_article-without-is_country was set during scanning.
    tokens_global = tokenize("Article 21", is_global=True)
    tokens_non_global = tokenize("Article 21", is_global=False)

    remapped = [t for t in tokens_global if t.query_text == "EXPERTS OPINION"]
    assert len(remapped) == 1
    assert remapped[0].proximity == 2
    assert remapped[0].group_id == "111050000000000051"
    # type is NOT rewritten - whatever KeyWord/SectionTypeFormat classification "Article"
    # itself received stays unchanged
    assert remapped[0].type != ""

    assert not any(t.query_text == "EXPERTS OPINION" for t in tokens_non_global)


def test_article_remap_does_not_fire_when_is_country_also_set():
    # cs:1970: `if (IsArticle && !IsCountry)` - a query that also classifies a Country
    # token suppresses the remap entirely, even under is_global=True. "AUSTRALIA" is a
    # real element_type "56" (Country) dictionary key (confirmed live in
    # repotaxmannapi_token_dictionary.json, 2026-09-06).
    tokens = tokenize("Article 21 Australia", is_global=True)
    assert not any(t.query_text == "EXPERTS OPINION" for t in tokens)
```

- [ ] **Step 2: Run tests to verify the first one fails**

Run: `uv run pytest packages/common/tests/test_repotaxmannapi_tokenizer.py -k article_remapped -v`
Expected: FAIL (`tokenize()` has no `is_global` parameter yet)

- [ ] **Step 3: Add `is_global` parameter, `is_article`/`is_country` tracking, and the remap pass**

In `_Analyzer.__init__`, add an `is_global: bool = False` parameter, store as
`self.is_global`; also initialize `self.is_article = False` and `self.is_country = False`
(mirroring cs:1553-1554's per-`ProcessorQuery`-call locals — since this Python port's
`_Analyzer` is constructed once per `tokenize()` call and `process_query()` is called
exactly once on it, per-instance attributes are the correct equivalent scope).

In `process_query`'s main loop, set `self.is_article = True` in exactly the 3 places
confirmed above:
- Inside the `elif element_code == ElementType.KEY_WORD_ONLY:` branch (current
  `repotaxmannapi_tokenizer.py:774-785), after the existing `_push_token` call, add:
  `if r_text.upper().startswith("ARTICLE"): self.is_article = True` (cs:1600 — this branch
  has no success/failure split, always applies).
- Inside the `elif element_code == ElementType.KEY_WORD:` branch's `elif r_text.upper() !=
  "AS":` sub-branch ONLY (current lines ~799-808 — the `ProcessKeyWord` FAILURE path, not
  the `if ok:` success path above it), add the same check (cs:1621).
- Inside the `elif element_code == ElementType.KEY_WORD_TYPE2:` branch's `else:` sub-branch
  ONLY (current lines ~828-841 — the `ProcessKeyWordType2` FAILURE path, not the `if ok:`
  success path above it), add the same check (cs:1645).

Set `self.is_country = True` unconditionally inside the `elif element_code ==
ElementType.COUNTRY:` branch (current lines ~900-909; cs:~1697).

After the existing phrase-word tail-append loop (`repotaxmannapi_tokenizer.py:1149-1157`'s
current end, right before `return tokens`), add:

```python
        if self.is_global and self.is_article and not self.is_country:
            # TaxmannQueryAnalizer.cs:1968-1984: rewrite QueryText/Proximity (NOT Type) to
            # the "EXPERTSOPINION" dictionary entry's own values for every token whose
            # QueryText starts with "ARTICLE", and reset that token's group_id to
            # EXPERTSOPINION's own group_id (this port's per-token equivalent of the real
            # source's query-level ReSetPrimaryTag("EXPERTSOPINION") - see this module's
            # own docstring on the per-token group_id deviation).
            expertsopinion_text = _get_key_search_text("EXPERTSOPINION")
            expertsopinion_proximity = _get_key_proximity("EXPERTSOPINION")
            expertsopinion_group_id = classify_token("EXPERTSOPINION")["group_id"]
            for i, tok in enumerate(tokens):
                if tok.query_text.upper().startswith("ARTICLE"):
                    tokens[i] = RepotaxmannapiToken(
                        query_text=expertsopinion_text, query_date=tok.query_date,
                        org_text=tok.org_text, type=tok.type, or_in=tok.or_in,
                        proximity=expertsopinion_proximity, group_id=expertsopinion_group_id,
                    )

        return tokens
```

Update `tokenize()`'s own signature to accept and forward `is_global`:

```python
def tokenize(query: str, is_global: bool = False) -> list[RepotaxmannapiToken]:
    """Entry point: full port of `new TaxmannQueryAnalizer(query)` followed by a single
    `ProcessorQuery()` call. `is_global` (2026-09-06): threads through the real source's
    `isGlobalSearch` field - gates the ARTICLE->EXPERTSOPINION remap pass
    (TaxmannQueryAnalizer.cs:1968-1984). Defaults to False so every existing caller keeps
    today's behavior unchanged; only `es_client.py`'s repotaxmannapi global-search path
    passes True."""
    analyzer = _Analyzer(query, is_global=is_global)
    return analyzer.process_query()
```

- [ ] **Step 4: Update `es_client.py`'s `tokenize(query)` call site**

In `_build_repotaxmannapi_field_query`, change `tokens = tokenize(query)` to `tokens =
tokenize(query, is_global=True)` (this function only ever builds the global-search query —
confirm no other caller of `tokenize()` in this repo needs `is_global=True` before making
this the only call site that passes it; every other existing caller stays on the `False`
default, unchanged).

- [ ] **Step 5: Run both tests to verify they pass**

Run: `uv run pytest packages/common/tests/test_repotaxmannapi_tokenizer.py -k "article_remapped or article_remap_does_not_fire" -v`
Expected: PASS

- [ ] **Step 6: Run the whole common suite**

Run: `uv run pytest packages/common/tests -q`
Expected: only the one pre-existing unrelated failure — pay attention to whether any
existing `es_client.py`-level test asserts on token query_text/group_ids for an
`Article ...` query, since this change could shift its `group_id` resolution (which
`build_function_score_functions` consumes).

- [ ] **Step 7: Commit**

```bash
git add packages/common/src/common/repotaxmannapi_tokenizer.py packages/common/src/common/es_client.py packages/common/tests/test_repotaxmannapi_tokenizer.py
git commit -m "feat(repotaxmannapi): port ARTICLE->EXPERTSOPINION token remap for global search"
```

---

## Final check: whole-repo sanity pass

- [ ] Run `uv run pytest packages/common/tests -q` — only the one pre-existing unrelated
  failure (`test_build_query_preview_matches_what_raw_search_actually_sends`) should remain.
- [ ] Run `uv run pytest packages/retrieval-api/tests -q` (scoped, not the whole-repo bare
  `uv run pytest` — this repo's own CLAUDE.md convention) to confirm nothing in the
  retrieval-api package broke from the `build_should_clauses` return-shape change (Task 2)
  or the new `tokenize()` parameter (Task 8).
- [ ] Re-run the earlier "section 52" / citation-lookup manual traces (via
  `/v1/query-analysis` or the Instant-mode trace panel) once the user's ES reindexing
  (mentioned as in-progress) completes, to visually confirm the corrected static group
  boosts (Task 1) and the AND-of-OR restructure (Task 2) actually change the top-N ranking
  in the expected direction against a live index — this is a manual verification step, not
  an automated test, since it depends on live data this plan's automated tests don't have
  access to.
