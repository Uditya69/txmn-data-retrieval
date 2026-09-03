# repotaxmannapi Exact-Replica Query/Boost Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a byte-exact Python replica of `repotaxmannapi`'s (a separate, read-only .NET
codebase) Instant-mode query tokenizer, query builder, and `Multiply`-mode scoring stack,
reachable behind an opt-in flag alongside the existing (unchanged, eval-verified) sum-mode
default.

**Architecture:** Three new, independent Python modules under `packages/common/src/common/`
(`repotaxmannapi_token_dictionary.json` + loader, `repotaxmannapi_tokenizer.py`,
`repotaxmannapi_query_builder.py`), wired into Instant mode's `raw_search` via a new
`boost_source: Literal["sum", "repotaxmannapi"]` parameter (default `"sum"` — zero change to
existing behavior until explicitly opted into). Nothing in this plan modifies or removes any
existing function; it is purely additive.

**Tech Stack:** Python 3.11, pytest + pytest-asyncio (existing test stack), no new
dependencies — the token dictionary ships as a plain JSON file, no XML parsing needed at
runtime (extraction happens once, offline, via a one-time script).

**Spec:** `docs/superpowers/specs/2026-09-01-repotaxmannapi-exact-replica-design.md`

## Global Constraints

- All work happens on branch `feature/repotaxmannapi-exact-replica` (already created off
  `dev`) — never commit this work directly to `dev`.
- This is a **known, deliberate override** of CLAUDE.md's hard rule 5 (no `boost_mode:
  "multiply"`) — the new code path is additive/opt-in only; the existing default (`sum` mode)
  must never change behavior as a side effect of any task in this plan.
- Every constant (boost weight, factor, modifier, subgroup id) must be copied verbatim from
  the real C# source with an exact file:line citation in the code comment — never approximated
  or "close enough."
- `repotaxmannapi` is read-only, uncompilable from this environment — verification is by
  careful reading + cross-checking against real ES documents (already available via
  `common.es_client.get_es_client`), not by running the original C# code.
- Every task ends with `uv run pytest <scoped path> -q` passing before commit — scope to the
  package under test, never the whole repo (`uv run pytest` with no path is too slow for this
  session's iteration loop).

---

## Task 1: Token dictionary extraction

**Files:**
- Create: `packages/common/scripts/extract_repotaxmannapi_token_dictionary.py`
- Create: `packages/common/src/common/data/repotaxmannapi_token_dictionary.json`
- Create: `packages/common/src/common/repotaxmannapi_token_dictionary.py`
- Test: `packages/common/tests/test_repotaxmannapi_token_dictionary.py`

**Interfaces:**
- Produces: `load_repotaxmannapi_token_dictionary() -> dict[str, TokenDictEntry]` where
  `TokenDictEntry` is a `TypedDict` with keys `element_type: str`, `tag_no: str`,
  `proximity: int`, `boost_factor: int`, `group_id: str`, `search_text: str | None`.

### Background (read this before writing code)

`repotaxmannapi/TaxmannAPI/TokenParserElastic.resx` is a 2646-entry XML resource dictionary.
Each `<data name="KEYWORD"><value>...</value></data>` entry's value has the shape:

```
ELEMENTTYPE:TAGNO;PROXIMITY;BOOSTFACTOR;GROUPID:SEARCHTEXT
```

Verified real examples (`repotaxmannapi/TaxmannAPI/TokenParserElastic.resx`):
- `"A"` → `"99:0;2;0;0:"` (a stop word — `ElementType.StopWord = "99"`)
- `"AAAR"` → `"77:0;2;0;0:Appellate Authority For Advance Ruling|AAAR"` (a Zone-type entry,
  `ElementType.Zone = "77"`, with a `|`-separated synonym search-text list)
- `"AAR"` → `"89:0;2;0;0:AAR"` (a Court-type entry, `ElementType.Court = "89"`)
- `"ACCOUNTINGSTANDARD"` → `"66:2;0;0;111050000000011660:ACCOUNTING STANDARD"` (a KeyWord-type
  entry, `ElementType.KeyWord = "66"`, carrying a real `groupID`)

`ELEMENTTYPE` is one of the codes in `TaxmannQueryAnalizer.cs`'s `ElementType` class
(lines 38-57): `Month="40"`, `Synonym="44"`, `KeyWordOnly="55"`, `Country="56"`,
`KeyWordType2="64"`, `KeyWordOrStopWord="65"`, `KeyWord="66"`, `KeyWordHighCourt="67"`,
`KeyWordDated="69"`, `Zone="77"`, `NumAlphaZone="87"`, `Journal="88"`, `Court="89"`,
`SBTM="90"`, `MultiWordStopWord="98"`, `StopWord="99"`.

The colon-delimited value is split in `SetPrimaryTag` (`TaxmannQueryAnalizer.cs:288-317`):
```csharp
KeyValue = GetResource(KeyText.ToUpper());      // the raw "77:0;2;0;0:..." string
KeyItems = KeyValue.Split(':');                 // ["77", "0;2;0;0", "..."]
KeyProperties = KeyItems[1].Split(';');          // ["0", "2", "0", "0"]
iTagNo = KeyProperties[0];
iProxiWords = Convert.ToInt32(KeyProperties[1]);
iBoostFactor = Convert.ToInt64(KeyProperties[2]);
iGroupID = KeyProperties[3].Trim();
```
Note `KeyItems[0]` (the leading `ELEMENTTYPE` code, e.g. `"77"`) is used to route which
`ElementType` branch a token falls into elsewhere in the tokenizer (Task 2 will confirm the
exact call sites), but is **not** the same as `iTagNo` (`KeyProperties[0]`, taken from inside
the middle segment) — keep both fields distinct in the extracted dictionary; do not conflate
them.

`GetKeySearchText` (`TaxmannQueryAnalizer.cs:360-376`) returns `KeyItems[2].Trim()` only when
`KeyItems.Length == 3` (i.e. the value has a search-text segment after the second colon) —
some entries (like `"A"` above) have an empty trailing segment; store `search_text: None` for
those, not an empty string, so `load_repotaxmannapi_token_dictionary()` callers can
distinguish "no search text" from "empty search text."

- [ ] **Step 1: Write the extraction script**

Create `packages/common/scripts/extract_repotaxmannapi_token_dictionary.py`:

```python
"""One-time offline extraction: repotaxmannapi/TaxmannAPI/TokenParserElastic.resx ->
packages/common/src/common/data/repotaxmannapi_token_dictionary.json.

Run manually whenever repotaxmannapi's TokenParserElastic.resx changes:
    uv run python packages/common/scripts/extract_repotaxmannapi_token_dictionary.py \
        /path/to/repotaxmannapi/TaxmannAPI/TokenParserElastic.resx

Not run automatically - repotaxmannapi is a separate, read-only checkout not guaranteed to
be present at a fixed path in every environment that runs this repo's tests. The generated
JSON is committed to this repo so tests never depend on repotaxmannapi being checked out.
"""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "src" / "common" / "data" / "repotaxmannapi_token_dictionary.json"
)


def parse_entry(name: str, raw_value: str) -> dict:
    """Parses one <data name="..."><value>ELEMENTTYPE:TAGNO;PROX;BOOST;GROUP:SEARCHTEXT
    </value></data> entry per TaxmannQueryAnalizer.cs's SetPrimaryTag/GetKeySearchText
    (TaxmannQueryAnalizer.cs:288-317, 360-376) - see this script's own module docstring
    reference and the plan task's "Background" section for the verified field mapping."""
    parts = raw_value.split(":")
    element_type = parts[0]
    properties = parts[1].split(";")
    search_text = parts[2].strip() if len(parts) == 3 and parts[2].strip() else None
    return {
        "element_type": element_type,
        "tag_no": properties[0],
        "proximity": int(properties[1]),
        "boost_factor": int(properties[2]),
        "group_id": properties[3].strip(),
        "search_text": search_text,
    }


def main(resx_path: str) -> None:
    tree = ET.parse(resx_path)
    root = tree.getroot()
    entries = {}
    for data_el in root.findall("data"):
        name = data_el.get("name")
        value_el = data_el.find("value")
        if name is None or value_el is None or value_el.text is None:
            continue
        # Skip the schema's own documentation examples (Name1/Color1/Bitmap1/Icon1 - see
        # the .resx header comment) - these aren't real token entries.
        if name in ("Name1", "Color1", "Bitmap1", "Icon1"):
            continue
        try:
            entries[name] = parse_entry(name, value_el.text)
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Failed to parse entry {name!r} = {value_el.text!r}") from exc

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT_PATH.write_text(json.dumps(entries, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {len(entries)} entries to {_OUTPUT_PATH}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
```

- [ ] **Step 2: Run the extraction script against the real resx file**

Run: `uv run python packages/common/scripts/extract_repotaxmannapi_token_dictionary.py "C:\A_VANSH\ALL_NEEDS\CODING\Taxmann\txmn-data-retrieval\repotaxmannapi\TaxmannAPI\TokenParserElastic.resx"`

Expected output: `Wrote 2642 entries to <path>` (2646 `<data>` elements minus the 4 schema
example entries skipped above - if the count differs, inspect why before proceeding, don't
just accept a different number).

- [ ] **Step 3: Write the failing test for the loader and known entries**

```python
# packages/common/tests/test_repotaxmannapi_token_dictionary.py
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
```

- [ ] **Step 4: Run test to verify it fails**

Run: `uv run pytest packages/common/tests/test_repotaxmannapi_token_dictionary.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'common.repotaxmannapi_token_dictionary'`

- [ ] **Step 5: Write the loader module**

```python
# packages/common/src/common/repotaxmannapi_token_dictionary.py
"""Loads the extracted repotaxmannapi token dictionary (see
scripts/extract_repotaxmannapi_token_dictionary.py for how data/
repotaxmannapi_token_dictionary.json is generated from repotaxmannapi's
TokenParserElastic.resx - a separate, read-only .NET codebase). Cached at module level:
the dictionary is ~2600 entries, small enough to load once and reuse across every call in a
process, same pattern common.legal_lexicon already uses for its own JSON-backed lexicon."""
import json
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

_DATA_PATH = Path(__file__).parent / "data" / "repotaxmannapi_token_dictionary.json"


class TokenDictEntry(TypedDict):
    element_type: str
    tag_no: str
    proximity: int
    boost_factor: int
    group_id: str
    search_text: str | None


@lru_cache(maxsize=1)
def load_repotaxmannapi_token_dictionary() -> dict[str, TokenDictEntry]:
    return json.loads(_DATA_PATH.read_text(encoding="utf-8"))
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest packages/common/tests/test_repotaxmannapi_token_dictionary.py -v`
Expected: PASS (all 4 tests)

- [ ] **Step 7: Commit**

```bash
git add packages/common/scripts/extract_repotaxmannapi_token_dictionary.py \
        packages/common/src/common/data/repotaxmannapi_token_dictionary.json \
        packages/common/src/common/repotaxmannapi_token_dictionary.py \
        packages/common/tests/test_repotaxmannapi_token_dictionary.py
git commit -m "feat(repotaxmannapi-replica): extract and load token dictionary from resx"
```

---

## Task 2: Tokenizer — stopword and element-type classification

**Files:**
- Create: `packages/common/src/common/repotaxmannapi_tokenizer.py`
- Test: `packages/common/tests/test_repotaxmannapi_tokenizer.py`

**Interfaces:**
- Consumes: `load_repotaxmannapi_token_dictionary()` (Task 1)
- Produces: `classify_token(word: str) -> TokenDictEntry | None` — looks up `word.upper()` in
  the dictionary, returns `None` for an unrecognized word (mirrors
  `TaxmannQueryAnalizer.cs`'s `GetResource` returning `""` for a missing key, which
  `SetPrimaryTag` then treats as "no entry" — `TaxmannQueryAnalizer.cs:296-314`).

### Background (read this before writing code)

Before any tokenization, `TaxmannQueryAnalizer`'s constructor (`TaxmannQueryAnalizer.cs:114-193`)
does these string-level transforms, in this exact order, on the raw query text:

1. Trim.
2. Collapse a hyphen surrounded by alphanumerics into a space:
   `Regex.Replace(text, @"(?<=[A-Za-z0-9])\s*-\s*(?=[A-Za-z0-9])", " ")` (line 126).
3. Replace `""` (empty double-quote pair) with a single space.
4. Extract every double-quoted phrase into `PhraseElements` (a separate list), removing it
   from the main text (lines 130-159) — this is a manual paired-quote scanner, not a regex;
   read lines 130-159 directly before porting, it has several edge cases (unterminated quote
   at end of string, quote with nothing after it).
5. Collapse repeated spaces to one.
6. Split on space into `QueryElement`.
7. For each element: strip any `'` character and everything after it up to the next space
   (`CheckAndRemove`, lines 194-216), then strip special characters (`WordStripSplChr` — find
   and read this method's own definition before porting; it is used but not shown in the
   excerpt read so far).
8. Repeatedly call `RemoveMultiWordStopWords()` until it returns false (line 185) — this
   method is not yet read; find and read it in full before porting Task 3 (it consumes
   `ElementType.MultiWordStopWord = "98"` entries, which are multi-word phrases stored as
   single dictionary keys).

- [ ] **Step 1: Write the failing test for the dictionary lookup wrapper**

```python
# packages/common/tests/test_repotaxmannapi_tokenizer.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/common/tests/test_repotaxmannapi_tokenizer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'common.repotaxmannapi_tokenizer'`

- [ ] **Step 3: Write the minimal implementation**

```python
# packages/common/src/common/repotaxmannapi_tokenizer.py
"""Python port of repotaxmannapi/TaxmannAPI/Elastic/TaxmannQueryAnalizer.cs (a separate,
read-only .NET codebase) - see docs/superpowers/specs/2026-09-01-repotaxmannapi-exact-replica-design.md
for why this exists as a parallel, opt-in path rather than replacing this repo's existing
(eval-verified, sum-mode) tokenizer."""
from common.repotaxmannapi_token_dictionary import TokenDictEntry, load_repotaxmannapi_token_dictionary


def classify_token(word: str) -> TokenDictEntry | None:
    """Direct port of TaxmannQueryAnalizer.cs's GetResource(word.ToUpper()) lookup
    (TaxmannQueryAnalizer.cs:217-233) - the C# returns "" for a missing key, which
    SetPrimaryTag (lines 296-314) treats as "no dictionary entry"; this returns None for
    the same case so callers use a Pythonic None-check instead of an empty-string check."""
    return load_repotaxmannapi_token_dictionary().get(word.upper())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/common/tests/test_repotaxmannapi_tokenizer.py -v`
Expected: PASS (all 3 tests)

- [ ] **Step 5: Commit**

```bash
git add packages/common/src/common/repotaxmannapi_tokenizer.py \
        packages/common/tests/test_repotaxmannapi_tokenizer.py
git commit -m "feat(repotaxmannapi-replica): add token dictionary classification wrapper"
```

---

## Task 3: Tokenizer — full `TaxmannQueryAnalizer` port (research + implementation)

**This task cannot be fully scripted in advance** — it requires reading
`TaxmannQueryAnalizer.cs` lines 194-2008 in full (the constructor's remaining helper methods
`WordStripSplChr`, `RemoveMultiWordStopWords`, plus the entire token-scanning loop that
builds `QueryTokens` from `QueryElement`, covering Month/Date/Citation/Notification/Zone/
NumAlphaZone detection) before any port code can be written faithfully. Attempting to
pre-write that port without having read those ~1800 lines would produce fabricated,
unverified code — exactly what this plan must not do.

**Files:**
- Modify: `packages/common/src/common/repotaxmannapi_tokenizer.py`
- Test: `packages/common/tests/test_repotaxmannapi_tokenizer.py`

**Interfaces:**
- Consumes: `classify_token` (Task 2)
- Produces: `tokenize(query: str) -> list[RepotaxmannapiToken]` where `RepotaxmannapiToken` is
  a dataclass mirroring the C# `Token` struct (`TaxmannQueryAnalizer.cs:84-92`): `query_text:
  str`, `query_date: date | None`, `org_text: str`, `type: str`, `or_in: bool`,
  `proximity: int`.

- [ ] **Step 1: Read `TaxmannQueryAnalizer.cs` lines 194-700** (constructor helpers +
  start of the main tokenization loop). Take notes on: `WordStripSplChr`'s exact character
  set, `RemoveMultiWordStopWords`'s multi-word lookahead algorithm, and the first token-type
  branch (`TokenType.PhraseWord` handling for the `PhraseElements` extracted earlier).

- [ ] **Step 2: Read `TaxmannQueryAnalizer.cs` lines 700-1400** (Month/Date/Citation/
  Notification detection branches). Take notes on each regex/format string used, verbatim.

- [ ] **Step 3: Read `TaxmannQueryAnalizer.cs` lines 1400-2008** (Zone/NumAlphaZone/
  remaining branches, plus `getAnalyzeSearch`'s final assembly into `queryList`/`query`).

- [ ] **Step 4: Write failing tests for each token-type branch identified in Steps 1-3**,
  one test per branch, each asserting on a query string this plan's author (or the task
  executor) has traced through the C# source by hand to determine the expected token
  sequence — not guessed. Do not proceed to Step 5 until every branch found in Steps 1-3 has
  at least one such test.

- [ ] **Step 5: Implement `tokenize()`** to make each test from Step 4 pass, one branch at a
  time (red-green per branch, not one big implementation at the end).

- [ ] **Step 6: Run the full tokenizer test file and verify all tests pass.**

Run: `uv run pytest packages/common/tests/test_repotaxmannapi_tokenizer.py -v`

- [ ] **Step 7: Commit**

```bash
git add packages/common/src/common/repotaxmannapi_tokenizer.py \
        packages/common/tests/test_repotaxmannapi_tokenizer.py
git commit -m "feat(repotaxmannapi-replica): port full TaxmannQueryAnalizer tokenization loop"
```

---

## Task 4: Query builder — phrase boosts and field structure

**Files:**
- Create: `packages/common/src/common/repotaxmannapi_query_builder.py`
- Test: `packages/common/tests/test_repotaxmannapi_query_builder.py`

**Interfaces:**
- Consumes: `tokenize()` (Task 3)
- Produces: `build_should_clauses(tokens: list[RepotaxmannapiToken], is_global: bool, is_excus: bool) -> list[dict]`

### Background (read this before writing code)

Already verified in this session (via a real pasted low-level-query trace, cross-checked
against `SearchTextElastic.cs:279-292` and `GlobalSearchResearch.cs`'s
`getSearchResult`/`getLowLevelQuery` output): the real phrase-boost tiers, exactly as sent to
ES for a `GetSearchResult` (Global search) call:

```
heading:        boost 155000, analyzer "snowball", slop from token.Proximity
subheading:     boost  80000, analyzer "snowball"
searchboosttext: boost 70000, analyzer "snowball"  (non-Excus path)
headnotestext:  boost  65000
fullcontent:    boost      5, analyzer "snowball"  (non-Excus path)
```
When `isExcus == true` (a `PH`-type token sets this — `SearchTextElastic.cs:255-258`), the
field suffix changes to `.phrase_search` and the `fullcontent`/`subheading` boost values stay
the same but the `analyzer` option is dropped (see `SearchTextElastic.cs:279-292` for both
branches side by side).

A `SECTION `-prefixed token (checked via `el.IndexOf("SECTION ") > -1`,
`SearchTextElastic.cs:295, 325`) additionally adds a `fullcontent` (or
`fullcontent.phrase_search` under Excus) minus-clause for `"SUB " + el"` at boost 5 — this
excludes documents where the section reference is actually a sub-section back-reference. Port
this exactly; do not drop it as a minor detail — it is a real, deliberate exclusion rule.

- [ ] **Step 1: Write the failing test for the non-Excus, non-SECTION case**

```python
# packages/common/tests/test_repotaxmannapi_query_builder.py
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
        "searchboosttext": 70000, "headnotestext": 65000, "fullcontent": 5,
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/common/tests/test_repotaxmannapi_query_builder.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the minimal `build_should_clauses` for this one case**

```python
# packages/common/src/common/repotaxmannapi_query_builder.py
"""Python port of repotaxmannapi/TaxmannAPI/Elastic/SearchTextElastic.cs's GetQuery
(a separate, read-only .NET codebase) - phrase-boost tiers and field names verified
2026-09-01 against both the C# source and a real captured low-level-query trace from
GlobalSearchResearch.cs's own GetSearchResult/GetLowLevelQuery output."""

_PHRASE_BOOSTS_STANDARD = {
    "heading": 155000, "subheading": 80000,
    "searchboosttext": 70000, "headnotestext": 65000, "fullcontent": 5,
}


def build_should_clauses(tokens: list, is_global: bool, is_excus: bool) -> list[dict]:
    should = []
    for token in tokens:
        for field, boost in _PHRASE_BOOSTS_STANDARD.items():
            field_name = f"{field}.phrase_search" if is_excus else field
            clause = {"match_phrase": {field_name: {"query": token.query_text, "boost": boost, "slop": token.proximity}}}
            if not is_excus and field in ("heading", "subheading", "searchboosttext", "fullcontent"):
                clause["match_phrase"][field_name]["analyzer"] = "snowball"
            should.append(clause)
    return should
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/common/tests/test_repotaxmannapi_query_builder.py -v`
Expected: PASS

- [ ] **Step 5: Write the failing test for the SECTION-prefixed minus-clause case**

```python
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
    assert minus_clauses[0]["match_phrase"]["fullcontent"]["boost"] == 5
```

- [ ] **Step 6: Run test to verify it fails**

Run: `uv run pytest packages/common/tests/test_repotaxmannapi_query_builder.py -v`
Expected: FAIL (no minus-clause is generated yet)

- [ ] **Step 7: Implement the SECTION-prefix minus-clause**, reading
  `SearchTextElastic.cs:293-334` first to confirm the exact condition (`element.Type == "T1"
  && el.IndexOf("SECTION ") > -1` for the non-`isOver` branch, per the earlier excerpt) before
  writing the code:

```python
def build_should_clauses(tokens: list, is_global: bool, is_excus: bool) -> list[dict]:
    should = []
    for token in tokens:
        field_suffix = ".phrase_search" if is_excus else ""
        for field, boost in _PHRASE_BOOSTS_STANDARD.items():
            field_name = f"{field}{field_suffix}"
            clause = {"match_phrase": {field_name: {"query": token.query_text, "boost": boost, "slop": token.proximity}}}
            if not is_excus and field in ("heading", "subheading", "searchboosttext", "fullcontent"):
                clause["match_phrase"][field_name]["analyzer"] = "snowball"
            should.append(clause)
        if token.type == "T1" and "SECTION " in token.query_text:
            fullcontent_field = f"fullcontent{field_suffix}"
            minus_clause = {
                "match_phrase": {
                    fullcontent_field: {
                        "query": f"SUB {token.query_text}", "boost": 5, "slop": token.proximity,
                    },
                },
            }
            if not is_excus:
                minus_clause["match_phrase"][fullcontent_field]["analyzer"] = "snowball"
            should.append(minus_clause)
    return should
```

- [ ] **Step 8: Run both tests to verify they pass**

Run: `uv run pytest packages/common/tests/test_repotaxmannapi_query_builder.py -v`
Expected: PASS (both tests)

- [ ] **Step 9: Commit**

```bash
git add packages/common/src/common/repotaxmannapi_query_builder.py \
        packages/common/tests/test_repotaxmannapi_query_builder.py
git commit -m "feat(repotaxmannapi-replica): port phrase-boost and SUB-exclusion query building"
```

---

## Task 5: Query builder — remaining `GetQuery` branches (research + implementation)

**Same caveat as Task 3** — `SearchTextElastic.cs`'s `GetQuery` is ~700 lines
(`SearchTextElastic.cs:207-900`, per this session's earlier read) covering date-format
tokens (`MonthFmt`/`DateMonthFmt`), the `isOver`/`OR`-group handling for pipe-separated
synonym alternatives, and citation assembly. This cannot be pre-written without reading it in
full.

**Files:**
- Modify: `packages/common/src/common/repotaxmannapi_query_builder.py`
- Test: `packages/common/tests/test_repotaxmannapi_query_builder.py`

- [ ] **Step 1: Read `SearchTextElastic.cs` lines 340-900** in full (the remainder of
  `GetQuery` not yet covered by Task 4's background section).

- [ ] **Step 2: Write one failing test per remaining branch identified in Step 1**, each
  traced by hand against the C# source for its expected output — same discipline as Task 3
  Step 4.

- [ ] **Step 3: Implement each branch** to make its test pass, one at a time.

- [ ] **Step 4: Run the full query-builder test file and verify all tests pass.**

Run: `uv run pytest packages/common/tests/test_repotaxmannapi_query_builder.py -v`

- [ ] **Step 5: Commit**

```bash
git add packages/common/src/common/repotaxmannapi_query_builder.py \
        packages/common/tests/test_repotaxmannapi_query_builder.py
git commit -m "feat(repotaxmannapi-replica): port remaining GetQuery branches"
```

---

## Task 6: Scoring — recency ladder and static field boosts

**Files:**
- Create: `packages/common/src/common/repotaxmannapi_scoring.py`
- Test: `packages/common/tests/test_repotaxmannapi_scoring.py`

**Interfaces:**
- Produces: `build_function_score_functions(group_id: str) -> list[dict]`

### Background (already verified in this session — see
`docs/superpowers/specs/2026-09-01-repotaxmannapi-exact-replica-design.md`'s Scope section
and `GlobalSearchResearch.cs:607-653` read earlier)

Real recency ladder (8 tiers, `GlobalSearchResearch.cs:627-634`):
```
now-1d  to now:      weight 18
now-7d  to now-1d:   weight 15
now-1M  to now-7d:   weight 13
now-3M  to now-1M:   weight 10
now-1y  to now-3M:   weight 8
now-2y  to now-1y:   weight 5
now-5y  to now-2y:   weight 3.5
now-150y to now-5y:  weight 1.5
```

Real field_value_factor stack (`GlobalSearchResearch.cs:649-653`):
```
documenttypeboost:  factor 1 (NEST default - no .Factor() call), no modifier
viewcount:           factor 0.0000018, modifier log2p
court_boost:         factor 0.0000018, modifier log2p, missing 0
total_score:         factor 0.0000018, modifier log2p, missing 0
landmarkruling:      factor 1.2, modifier log2p, missing 0, filtered to exclude landmarkruling == -10
```

Note this is a **different** `court_boost`/`documenttypeboost` factor than this repo's
existing sum-mode `_apply_boost` (`court_boost` factor `0.01`/no modifier there vs `0.0000018`/
`log2p` here; `documenttypeboost` factor `0.2`/`sqrt` there vs `1`/none here) — this is
expected and correct: they are two different, independently-tuned formulas, not a bug to
reconcile. Do not "fix" one to match the other.

- [ ] **Step 1: Write the failing test for the recency ladder**

```python
# packages/common/tests/test_repotaxmannapi_scoring.py
from common.repotaxmannapi_scoring import build_function_score_functions


def test_recency_ladder_matches_real_8_tier_formula():
    functions = build_function_score_functions(group_id="0")
    recency = [
        fn for fn in functions
        if "filter" in fn and "range" in fn.get("filter", {})
        and "formatteddocumentdate" in fn["filter"]["range"]
    ]
    assert len(recency) == 8
    weights = sorted((fn["weight"] for fn in recency), reverse=True)
    assert weights == [18, 15, 13, 10, 8, 5, 3.5, 1.5]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/common/tests/test_repotaxmannapi_scoring.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the recency ladder + static field boosts**

```python
# packages/common/src/common/repotaxmannapi_scoring.py
"""Python port of repotaxmannapi/TaxmannAPI/Elastic/GlobalSearchResearch.cs's FunctionScore
stack (a separate, read-only .NET codebase). Every constant here is copied verbatim from
GlobalSearchResearch.cs:607-653, verified 2026-09-01 - see
docs/superpowers/specs/2026-09-01-repotaxmannapi-exact-replica-design.md.

This module intentionally uses different constants than common.es_client's own
_apply_boost (this repo's existing sum-mode formula) for the same-named fields
(court_boost, documenttypeboost) - the two are independently-tuned formulas for two
different combination modes (multiply here, sum there), not a discrepancy to reconcile."""

_RECENCY_TIERS = [
    ("now-1d", "now", 18),
    ("now-7d", "now-1d", 15),
    ("now-1M", "now-7d", 13),
    ("now-3M", "now-1M", 10),
    ("now-1y", "now-3M", 8),
    ("now-2y", "now-1y", 5),
    ("now-5y", "now-2y", 3.5),
    ("now-150y", "now-5y", 1.5),
]


def build_function_score_functions(group_id: str) -> list[dict]:
    functions = [
        {"filter": {"range": {"formatteddocumentdate": {"gte": gte, "lte": lte}}}, "weight": weight}
        for gte, lte, weight in _RECENCY_TIERS
    ]
    functions.append({"field_value_factor": {"field": "documenttypeboost"}})
    functions.append({"field_value_factor": {"field": "viewcount", "factor": 0.0000018, "modifier": "log2p"}})
    functions.append({
        "field_value_factor": {"field": "court_boost", "factor": 0.0000018, "modifier": "log2p", "missing": 0},
    })
    functions.append({
        "field_value_factor": {"field": "total_score", "factor": 0.0000018, "modifier": "log2p", "missing": 0},
    })
    functions.append({
        "filter": {"bool": {"must_not": [{"term": {"landmarkruling": -10}}]}},
        "field_value_factor": {"field": "landmarkruling", "factor": 1.2, "modifier": "log2p", "missing": 0},
    })
    return functions
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/common/tests/test_repotaxmannapi_scoring.py -v`
Expected: PASS

- [ ] **Step 5: Write the failing test for the field_value_factor stack**

```python
def test_field_value_factor_stack_matches_real_formula():
    functions = build_function_score_functions(group_id="0")
    by_field = {
        fn["field_value_factor"]["field"]: fn["field_value_factor"]
        for fn in functions if "field_value_factor" in fn
    }
    assert by_field["documenttypeboost"] == {"field": "documenttypeboost"}
    assert by_field["viewcount"] == {"field": "viewcount", "factor": 0.0000018, "modifier": "log2p"}
    assert by_field["court_boost"] == {
        "field": "court_boost", "factor": 0.0000018, "modifier": "log2p", "missing": 0,
    }
    assert by_field["total_score"] == {
        "field": "total_score", "factor": 0.0000018, "modifier": "log2p", "missing": 0,
    }
    assert by_field["landmarkruling"] == {
        "field": "landmarkruling", "factor": 1.2, "modifier": "log2p", "missing": 0,
    }
```

- [ ] **Step 6: Run test to verify it passes** (implementation already covers this from
  Step 3)

Run: `uv run pytest packages/common/tests/test_repotaxmannapi_scoring.py -v`
Expected: PASS (both tests)

- [ ] **Step 7: Commit**

```bash
git add packages/common/src/common/repotaxmannapi_scoring.py \
        packages/common/tests/test_repotaxmannapi_scoring.py
git commit -m "feat(repotaxmannapi-replica): port recency ladder and field_value_factor stack"
```

---

## Task 7: Scoring — groupBoost resolution and edition subgroup boosts

**Files:**
- Modify: `packages/common/src/common/repotaxmannapi_scoring.py`
- Test: `packages/common/tests/test_repotaxmannapi_scoring.py`

### Background (already verified in this session)

`GlobalSearchResearch.cs:607-615`:
```csharp
int groupBoost = 10000000;
if (groupid == Constants_GetIdByName.ActGroupId) groupBoost = 2;
if (groupid == Constants_GetIdByName.RuleFormId) groupBoost = 4;
```
where `Constants_GetIdByName.ActGroupId = "111050000000000064"` and
`Constants_GetIdByName.RuleFormId = "111050000000000026"` (confirmed against
`repotaxmannapi/TaxmannAPI/BL/Constants.cs:292,313`, read earlier this session).

`GlobalSearchResearch.cs:620-625` (weight functions using `groupid`/`groupBoost`):
```csharp
.Weight(w => w.Filter(fi => fi.Match(r => r.Field(d => d.groups.group.id).Query(groupid))).Weight(groupBoost))
.Weight(w0 => w0.Filter(fi => fi.Match(r => r.Field(d => d.groups.group.subgroup.id).Query(groupid))).Weight(groupBoost))
.Weight(w1 => w1.Filter(fi => fi.Match(r => r.Field(d => d.groups.group.subgroup.id).Query(
    groupid == ActGroupId ? IncomeTaxAct1961SGroupId : groupid == RuleFormId ? IncomeTaxRule1962 : null
))).Weight(2))
.Weight(wc => wc.Filter(fi => fi.Match(r => r.Field(d => d.groups.group.subgroup.id).Query(
    groupid == ActGroupId ? IncomeTaxAct2025SGroupId : groupid == RuleFormId ? IncomeTaxRule2026 : null
))).Weight(3))
```
`IncomeTaxAct1961SGroupId = "111050000000010687"`, `IncomeTaxAct2025SGroupId =
"111050000000020042"` (both already verified live against this repo's own ES index earlier
this session — see `common.es_client._EDITION_BOOSTS_BY_INSTRUMENT_KIND`, the sum-mode
version of this same concept). `IncomeTaxRule1962`/`IncomeTaxRule2026` in `Constants.cs` use
a `103010...`-pattern id that does **not** match `groups.group.subgroup.id`'s real id
namespace (confirmed via live ES query earlier — the real subgroup id for "Income-tax Rules,
1962" is `111050000000010121`, for "...2026" is `111050000000020129`). Use the verified
`111050...` ids here, not the `Constants.cs` ones, and note this discrepancy in the code
comment (this is a genuine bug/inconsistency in the source being replicated, not something to
silently "fix" by guessing — the plan makes an explicit choice: use the value that live-verified
against the real index, since a `Match` query against a wrong id would never match anything).

- [ ] **Step 1: Write the failing test for the groupBoost value resolution**

```python
def test_group_boost_defaults_to_ten_million_for_unrelated_group():
    functions = build_function_score_functions(group_id="999999999")
    group_id_fn = next(
        fn for fn in functions
        if fn.get("filter", {}).get("match", {}).get("groups.group.id", {}).get("query") == "999999999"
    )
    assert group_id_fn["weight"] == 10000000


def test_group_boost_is_two_for_act_group():
    functions = build_function_score_functions(group_id="111050000000000064")
    group_id_fn = next(
        fn for fn in functions
        if fn.get("filter", {}).get("match", {}).get("groups.group.id", {}).get("query") == "111050000000000064"
    )
    assert group_id_fn["weight"] == 2


def test_group_boost_is_four_for_rule_group():
    functions = build_function_score_functions(group_id="111050000000000026")
    group_id_fn = next(
        fn for fn in functions
        if fn.get("filter", {}).get("match", {}).get("groups.group.id", {}).get("query") == "111050000000000026"
    )
    assert group_id_fn["weight"] == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/common/tests/test_repotaxmannapi_scoring.py -v`
Expected: FAIL (no `groups.group.id` match function exists yet)

- [ ] **Step 3: Implement groupBoost resolution and the four group/edition weight
  functions**

```python
_ACT_GROUP_ID = "111050000000000064"
_RULE_FORM_ID = "111050000000000026"
# Verified live against this repo's own ES index 2026-09-01 (see
# common.es_client._EDITION_BOOSTS_BY_INSTRUMENT_KIND for the sum-mode equivalent lookup) -
# NOT the ids in repotaxmannapi's own BL/Constants.cs IncomeTaxRule1962/2026 constants,
# which use a different id namespace (a `rule` associate id, not groups.group.subgroup.id)
# and would never match a real document under this field.
_INCOME_TAX_ACT_1961_SUBGROUP_ID = "111050000000010687"
_INCOME_TAX_ACT_2025_SUBGROUP_ID = "111050000000020042"
_INCOME_TAX_RULES_1962_SUBGROUP_ID = "111050000000010121"
_INCOME_TAX_RULES_2026_SUBGROUP_ID = "111050000000020129"


def _resolve_group_boost(group_id: str) -> int:
    if group_id == _ACT_GROUP_ID:
        return 2
    if group_id == _RULE_FORM_ID:
        return 4
    return 10000000


def _resolve_edition_subgroup_id(group_id: str, *, current: bool) -> str | None:
    if group_id == _ACT_GROUP_ID:
        return _INCOME_TAX_ACT_2025_SUBGROUP_ID if current else _INCOME_TAX_ACT_1961_SUBGROUP_ID
    if group_id == _RULE_FORM_ID:
        return _INCOME_TAX_RULES_2026_SUBGROUP_ID if current else _INCOME_TAX_RULES_1962_SUBGROUP_ID
    return None


def build_function_score_functions(group_id: str) -> list[dict]:
    group_boost = _resolve_group_boost(group_id)
    functions = [
        {"filter": {"match": {"groups.group.id": {"query": group_id}}}, "weight": group_boost},
        {"filter": {"match": {"groups.group.subgroup.id": {"query": group_id}}}, "weight": group_boost},
    ]
    old_edition_id = _resolve_edition_subgroup_id(group_id, current=False)
    if old_edition_id is not None:
        functions.append({
            "filter": {"match": {"groups.group.subgroup.id": {"query": old_edition_id}}}, "weight": 2,
        })
    current_edition_id = _resolve_edition_subgroup_id(group_id, current=True)
    if current_edition_id is not None:
        functions.append({
            "filter": {"match": {"groups.group.subgroup.id": {"query": current_edition_id}}}, "weight": 3,
        })
    functions.extend([
        {"filter": {"range": {"formatteddocumentdate": {"gte": gte, "lte": lte}}}, "weight": weight}
        for gte, lte, weight in _RECENCY_TIERS
    ])
    functions.append({"field_value_factor": {"field": "documenttypeboost"}})
    functions.append({"field_value_factor": {"field": "viewcount", "factor": 0.0000018, "modifier": "log2p"}})
    functions.append({
        "field_value_factor": {"field": "court_boost", "factor": 0.0000018, "modifier": "log2p", "missing": 0},
    })
    functions.append({
        "field_value_factor": {"field": "total_score", "factor": 0.0000018, "modifier": "log2p", "missing": 0},
    })
    functions.append({
        "filter": {"bool": {"must_not": [{"term": {"landmarkruling": -10}}]}},
        "field_value_factor": {"field": "landmarkruling", "factor": 1.2, "modifier": "log2p", "missing": 0},
    })
    return functions
```

(This replaces the Task 6 `build_function_score_functions` body wholesale — update the
existing function in place rather than defining it twice.)

- [ ] **Step 4: Run all scoring tests to verify they pass**

Run: `uv run pytest packages/common/tests/test_repotaxmannapi_scoring.py -v`
Expected: PASS (all tests from Task 6 and Task 7)

- [ ] **Step 5: Commit**

```bash
git add packages/common/src/common/repotaxmannapi_scoring.py \
        packages/common/tests/test_repotaxmannapi_scoring.py
git commit -m "feat(repotaxmannapi-replica): port groupBoost resolution and edition subgroup boosts"
```

---

## Task 8: Scoring — the two multiply-mode penalty functions

**Files:**
- Modify: `packages/common/src/common/repotaxmannapi_scoring.py`
- Test: `packages/common/tests/test_repotaxmannapi_scoring.py`

### Background (already verified in this session)

`GlobalSearchResearch.cs:636-648`:
```csharp
.Filter(fq => { /* stateGstCatFilter AND caseLawsFilter (NOT caselaws group) */ }).Weight(0.03)
.Weight(w10 => w10.Filter(fq2 => { /* financeactBoostNewFilter AND NOT financeactBoostNewYearFilter */ }).Weight(0.02))
```
`stateGstCatFilter` = `categories.subcategory.id` in `Constants_GetIdByName.StateGSTCatID`.
`caseLawsFilter` = `groups.group.url` != `"caselaws"`.
`financeactBoostNewFilter` = `groups.group.subgroup.id` in `Constants_GetIdByName.FinanceActsSGroupId`.
`financeactBoostNewYearFilter` = `year.name` in the current `LattestFinanceActYearID` app
setting (an environment-config value in `repotaxmannapi`, not a compile-time constant — the
port must accept this as a parameter, not hardcode a year).

These are only meaningful under `boost_mode: "multiply"` (a near-zero weight crushes a doc's
score) — this is exactly why `build_function_score_functions` must only ever be assembled
into a `function_score` with `"score_mode": "multiply", "boost_mode": "multiply"` (Task 9),
never mixed into this repo's existing sum-mode `_apply_boost` output.

- [ ] **Step 1: Look up the real ids** for `Constants_GetIdByName.StateGSTCatID` and
  `Constants_GetIdByName.FinanceActsSGroupId` in `repotaxmannapi/TaxmannAPI/BL/Constants.cs`
  (grep for both names) and record them.

- [ ] **Step 2: Write the failing test**

```python
def test_state_gst_non_caselaws_penalty_present():
    functions = build_function_score_functions(group_id="0", latest_finance_act_year="2025")
    penalty = next(
        fn for fn in functions
        if fn.get("weight") == 0.03
    )
    assert "must_not" in penalty["filter"]["bool"]


def test_finance_act_old_year_penalty_present():
    functions = build_function_score_functions(group_id="0", latest_finance_act_year="2025")
    penalty = next(
        fn for fn in functions
        if fn.get("weight") == 0.02
    )
    assert penalty["filter"] is not None
```

(Fill in the exact `filter` shape assertions using the real ids found in Step 1 — do not
leave the ids as a placeholder in the final test.)

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest packages/common/tests/test_repotaxmannapi_scoring.py -v`
Expected: FAIL (no `0.03`/`0.02` weight functions exist yet)

- [ ] **Step 4: Implement both penalty functions**, adding a
  `latest_finance_act_year: str` parameter to `build_function_score_functions`, using the
  real ids recorded in Step 1.

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest packages/common/tests/test_repotaxmannapi_scoring.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add packages/common/src/common/repotaxmannapi_scoring.py \
        packages/common/tests/test_repotaxmannapi_scoring.py
git commit -m "feat(repotaxmannapi-replica): port StateGST and Finance-Act penalty functions"
```

---

## Task 9: Wire-up — `boost_source` flag in Instant mode

**Files:**
- Modify: `packages/common/src/common/es_client.py`
- Modify: `packages/retrieval-api/src/retrieval_api/instant/search.py`
- Test: `packages/common/tests/test_es_client.py`
- Test: `packages/retrieval-api/tests/test_instant_search.py`

**Interfaces:**
- Consumes: `build_should_clauses` (Task 5), `build_function_score_functions` (Task 8)
- Produces: `raw_search(..., boost_source: Literal["sum", "repotaxmannapi"] = "sum")` — new
  keyword parameter on the existing `raw_search` function; default unchanged so every
  existing caller is unaffected.

- [ ] **Step 1: Write the failing test**

```python
# packages/common/tests/test_es_client.py (add to existing file)
@pytest.mark.asyncio
async def test_raw_search_boost_source_repotaxmannapi_uses_multiply_mode():
    client = FakeAsyncES(search_hits=[])

    await raw_search(client, "Dimension Data India section 92C", limit=20, boost=True, boost_source="repotaxmannapi")

    query = client.search_calls[0]
    assert "function_score" in query
    fs = query["function_score"]
    assert fs["score_mode"] == "multiply"
    assert fs["boost_mode"] == "multiply"


@pytest.mark.asyncio
async def test_raw_search_boost_source_defaults_to_sum_mode_unchanged():
    """The new parameter must not change any existing caller's behavior - this is the
    regression guard for the default value."""
    client = FakeAsyncES(search_hits=[])

    await raw_search(client, "exemption claim", limit=20, boost=True)

    fs = client.search_calls[0]["function_score"]
    assert fs["score_mode"] == "sum"
    assert fs["boost_mode"] == "sum"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/common/tests/test_es_client.py -k boost_source -v`
Expected: FAIL with `TypeError: raw_search() got an unexpected keyword argument 'boost_source'`

- [ ] **Step 3: Read `raw_search`'s current signature and body** in
  `packages/common/src/common/es_client.py` (search for `def raw_search`) to find the exact
  point where `_apply_boost` is called, so the new branch can be inserted without disturbing
  the existing sum-mode path.

- [ ] **Step 4: Implement the `boost_source` parameter**

Add near the top of `es_client.py` (alongside the other imports):
```python
from common.repotaxmannapi_query_builder import build_should_clauses
from common.repotaxmannapi_scoring import build_function_score_functions
from common.repotaxmannapi_tokenizer import tokenize
```

In `raw_search`'s signature, add `boost_source: str = "sum"`. Where `_apply_boost` is
currently called when `boost=True`, branch:
```python
if boost:
    if boost_source == "repotaxmannapi":
        tokens = tokenize(query)
        should = build_should_clauses(tokens, is_global=True, is_excus=False)
        group_id = "0"  # Task 10 resolves this from tokens; "0" (no group signal) for now
        field_query = {
            "function_score": {
                "query": {"bool": {"should": should, "minimum_should_match": 1}},
                "functions": build_function_score_functions(group_id, latest_finance_act_year="2025"),
                "score_mode": "multiply",
                "boost_mode": "multiply",
            },
        }
    else:
        field_query = _apply_boost(field_query, chunks)
```

(Adjust variable names to match whatever the surrounding function actually calls its local
variables — read Step 3's findings first, do not blindly paste this over existing code.)

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest packages/common/tests/test_es_client.py -k boost_source -v`
Expected: PASS (both tests)

- [ ] **Step 6: Run the full `es_client` test file to confirm no regression**

Run: `uv run pytest packages/common/tests/test_es_client.py -q`
Expected: same pass count as before this task, plus the 2 new tests (one pre-existing
unrelated failure, `test_build_query_preview_matches_what_raw_search_actually_sends`, is
expected and already tracked separately — confirm it's still the *only* failure, not a new
one).

- [ ] **Step 7: Commit**

```bash
git add packages/common/src/common/es_client.py packages/common/tests/test_es_client.py
git commit -m "feat(repotaxmannapi-replica): wire boost_source flag into raw_search"
```

---

## Task 10: Group-id resolution from tokens

**Files:**
- Modify: `packages/common/src/common/es_client.py` (the `group_id = "0"` placeholder from
  Task 9 Step 4)
- Test: `packages/common/tests/test_es_client.py`

### Background

`SearchTextElastic.cs:589-590`: `if (stext.iGroupID != "0") groupid = stext.iGroupID;` — the
tokenizer's own `iGroupID` (set by `SetPrimaryTag`/`ReSetPrimaryTag` as tokens are classified,
per Task 2/3's dictionary lookups) becomes the `groupid` used throughout scoring, unless it's
`"0"` (no signal), in which case `groupid` stays empty. Task 3's `tokenize()` must already
expose each token's resolved `group_id` (from the dictionary entry that classified it) for
this to work — if Task 3's `RepotaxmannapiToken` dataclass doesn't already carry a `group_id`
field, add it there first as a prerequisite fix to Task 3, not as new scope here.

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_raw_search_repotaxmannapi_resolves_group_id_from_tokens():
    client = FakeAsyncES(search_hits=[])

    await raw_search(client, "Rule 6", limit=20, boost=True, boost_source="repotaxmannapi")

    functions = client.search_calls[0]["function_score"]["functions"]
    group_id_fn = next(
        fn for fn in functions
        if "groups.group.id" in fn.get("filter", {}).get("match", {})
    )
    assert group_id_fn["filter"]["match"]["groups.group.id"]["query"] == "111050000000000026"
```

(This assumes the token dictionary's `RULE` entry resolves to `group_id
"111050000000000026"` — verify this against the extracted dictionary from Task 1 before
finalizing the assertion; if the real extracted value differs, use the real value, not this
placeholder.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/common/tests/test_es_client.py -k resolves_group_id -v`
Expected: FAIL (group_id is hardcoded to `"0"` from Task 9)

- [ ] **Step 3: Implement group-id resolution** — replace the `group_id = "0"` placeholder
  with logic that scans `tokens` for the first non-`"0"` `group_id`, mirroring
  `SearchTextElastic.cs:589-590`'s "first non-zero wins" behavior (read that line again to
  confirm it's actually "first" and not "last" before implementing — the C# variable is
  reassigned inside a conditional inside what may be a loop; verify the real iteration order
  from context around line 589 before committing to an implementation).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/common/tests/test_es_client.py -k resolves_group_id -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/common/src/common/es_client.py packages/common/tests/test_es_client.py
git commit -m "feat(repotaxmannapi-replica): resolve groupBoost group_id from tokenized query"
```

---

## Task 11: Eval comparison run

**Files:**
- No new source files — this task runs existing eval tooling with the new flag enabled.

- [ ] **Step 1: Find the eval runner** — read `evals/README.md` (or equivalent) to find how
  `evals/retrieval_cases.json` is normally run against Instant mode.

- [ ] **Step 2: Run the eval suite against the existing default (`boost_source="sum"`)** to
  get a baseline pass count for this exact branch/commit (numbers may have drifted since the
  last recorded CLAUDE.md figures given the edition-boost/viewcount/group-signal work already
  landed on `dev`).

- [ ] **Step 3: Run the eval suite against `boost_source="repotaxmannapi"`** (this may
  require a small, temporary script or a CLI flag addition to the eval runner if it doesn't
  already accept one — if so, add the minimal flag needed, following the eval runner's
  existing patterns, and note this as an additional small commit).

- [ ] **Step 4: Write up the comparison** (pass count, which specific queries flip which way)
  as a new file `evals/2026-09-01-repotaxmannapi-replica-comparison.md`, following this
  repo's existing eval-results documentation pattern (check `eval-results/` for the format
  prior runs used).

- [ ] **Step 5: Commit**

```bash
git add evals/2026-09-01-repotaxmannapi-replica-comparison.md
git commit -m "docs(repotaxmannapi-replica): record eval comparison, sum vs multiply mode"
```

- [ ] **Step 6: Report the comparison to the user** and stop — do not merge this branch to
  `dev`, do not change the default `boost_source` value, and do not remove the existing
  sum-mode path. Whether to promote this path to the default (or at all) is the user's
  decision once they've seen the real numbers, not something this plan decides on its own.

---

## Self-Review Notes

- **Spec coverage:** all 5 phases from the design spec map onto Tasks 1-2 (Phase 1), 2-5
  (Phase 2), 4-5 (Phase 3, folded into the same tasks as the tokenizer since they share the
  same "read before writing" caveat), 6-8 (Phase 4), 9-11 (Phase 5). No spec section is
  without a task.
- **Placeholder scan:** Tasks 3 and 5 are explicitly scoped as research-then-implement tasks
  rather than pre-written code, because pre-writing a "port" of ~2500 lines of C# without
  having read it would itself be a fabricated placeholder masquerading as real content — this
  is flagged in-line in both tasks' preambles, not hidden.
- **Type consistency:** `RepotaxmannapiToken` is defined once (Task 3) and consumed
  identically in Tasks 4/5/10; `TokenDictEntry` is defined once (Task 1) and consumed
  identically in Tasks 2/3; `build_function_score_functions`'s signature grows across Tasks
  6/7/8 (documented explicitly as "replaces wholesale" / "adds a parameter" at each step, not
  silently).
