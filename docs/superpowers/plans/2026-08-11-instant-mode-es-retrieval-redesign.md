# Instant Mode ES Retrieval Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix Instant mode's ES query so it searches fields that actually have data, ranks using
two real-but-unused boost fields, and applies a no-LLM, sub-1s query-shape classifier (citation /
provision / plain-text) built from a cleaned-up port of `centax-node`'s `token.js` lexicon.

**Architecture:** Three new/changed files in `packages/common/src/common`: a data-extraction
script that turns `centax-node/constants/token.js` into a typed JSON lexicon, a
`legal_lexicon.py` module that loads that data plus hand-written regex patterns, a
`query_tokenizer.py` module that ports `queryAnalyzer.js`'s procedural merge/backtrack rules and
exposes a `classify_query_shape()` function, and an updated `es_client.raw_search` that uses the
classifier's output to pick field boosts and wraps the query in a `function_score`.

**Tech Stack:** Python 3.11, `elasticsearch` async client (already a dependency), `pytest` +
`pytest-asyncio` (existing test stack), Node.js (used once, at data-extraction time only, not a
runtime dependency).

## Global Constraints

- No LLM/model calls anywhere in this path — total request budget stays well under 1s (per
  `docs/superpowers/specs/2026-08-11-instant-mode-es-retrieval-redesign-design.md`, Non-goals).
- No new filter/routing behavior in Instant mode — ranking only.
- No changes to AI Mode (`intent.py`, `filter_resolve.py`, `retrieve.py`) — out of scope.
- `query_embed` role / Milvus fusion rules from `CLAUDE.md` are untouched — this plan is ES-only.
- Follow existing test patterns in `packages/common/tests/test_es_client.py`
  (`FakeAsyncES`, `client.search(index=, query=, size=)` call shape) — do not change that
  call signature.

---

## File Structure

- Create: `scripts/extract_token_lexicon.py` — one-time (rerunnable) script, reads
  `centax-node/constants/token.js` via Node, classifies rows by `ZoneType`, writes
  `packages/common/src/common/data/legal_lexicon.json`.
- Create: `packages/common/src/common/data/legal_lexicon.json` — generated data file (courts,
  journals, stopwords, synonyms/normalizations), committed to git (not regenerated at runtime).
- Create: `packages/common/src/common/legal_lexicon.py` — loads the JSON once at import time,
  exposes lookup functions + regex constants (Layers A+B from the design doc).
- Create: `packages/common/tests/test_legal_lexicon.py`.
- Create: `packages/common/src/common/query_tokenizer.py` — procedural tokenizer rules +
  `classify_query_shape()` (Layer C from the design doc).
- Create: `packages/common/tests/test_query_tokenizer.py`.
- Modify: `packages/common/src/common/es_client.py:6-8,32-44` — field list, `function_score`,
  boost-profile wiring into `raw_search`.
- Modify: `packages/common/tests/test_es_client.py` — update `raw_search` tests for the new
  query shape.

---

### Task 1: Extract `token.js` into a typed JSON lexicon

**Files:**
- Create: `scripts/extract_token_lexicon.py`
- Create: `packages/common/src/common/data/legal_lexicon.json` (generated output, committed)

**Interfaces:**
- Produces: `packages/common/src/common/data/legal_lexicon.json` with this exact shape (consumed
  by Task 2):
```json
{
  "courts": ["HIGH COURT", "SUPREME COURT", "..."],
  "journals": ["ITR", "CTR", "..."],
  "stopwords": ["ABLE", "..."],
  "synonyms": {"ACIT": ["ASSISTANT COMMISSIONER INCOME TAX", "ACIT"], "...": ["..."]},
  "normalizations": {"115I": "115-I", "...": "..."}
}
```

- [ ] **Step 1: Write the extraction script**

```python
"""One-time (rerunnable) extraction of centax-node's constants/token.js into a typed
JSON lexicon for this repo. Node.js is required only to run this script - it is not a
runtime dependency of the retrieval-system service.

Classification rule (by ZoneType, the human-assigned label already in token.js):
- ZoneType contains "COURT" or "BENCH" -> courts (both are judicial-body names; token.js
  itself uses BENCH for tribunal-city entries like "AAR"/"AHMEDABAD").
- ZoneType == "JOURNAL" -> journals.
- ZoneType == "STOPWORD" -> stopwords (SearchText is always empty for these).
- Everything else with a non-empty SearchText that differs from the key: if SearchText
  contains "|", it's a synonym/acronym-expansion list (split on "|", trim); otherwise it's
  a direct normalization (e.g. "115I" -> "115-I").
- Rows with empty SearchText equal to the key, or ZoneType "KEYWORD / COUNTRY" (country
  names aren't useful for query-shape boosting here), are dropped.
"""
import json
import subprocess
from pathlib import Path

TOKEN_JS_PATH = "/Users/uditya/dev/taxmann/centax-node/constants/token.js"
OUTPUT_PATH = Path(__file__).parent.parent / "packages/common/src/common/data/legal_lexicon.json"


def _load_token_json() -> dict:
    """Run token.js under Node and dump its TOKEN_JSON object as JSON on stdout.
    token.js has a trailing comma before its closing brace (valid JS, invalid JSON) and
    an unused `appObj` parameter, so it must be evaluated by Node, not JSON.parse'd."""
    script = (
        f"const configLoader = require('{TOKEN_JS_PATH}');"
        "const cfg = configLoader({});"
        "process.stdout.write(JSON.stringify(cfg.TOKEN_JSON));"
    )
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def build_lexicon(token_json: dict) -> dict:
    courts: set[str] = set()
    journals: set[str] = set()
    stopwords: set[str] = set()
    synonyms: dict[str, list[str]] = {}
    normalizations: dict[str, str] = {}

    for key, rows in token_json.items():
        row = rows[0]
        zone_type = row["ZoneType"]
        search_text = row["SearchText"].strip()

        if "COURT" in zone_type or "BENCH" in zone_type:
            courts.add(key)
        elif zone_type == "JOURNAL":
            journals.add(key)
        elif zone_type == "STOPWORD":
            stopwords.add(key)
        elif "COUNTRY" in zone_type:
            continue
        elif search_text and "|" in search_text:
            synonyms[key] = [part.strip() for part in search_text.split("|") if part.strip()]
        elif search_text and search_text.upper() != key.upper():
            normalizations[key] = search_text

    return {
        "courts": sorted(courts),
        "journals": sorted(journals),
        "stopwords": sorted(stopwords),
        "synonyms": dict(sorted(synonyms.items())),
        "normalizations": dict(sorted(normalizations.items())),
    }


def main() -> None:
    token_json = _load_token_json()
    lexicon = build_lexicon(token_json)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(lexicon, indent=2, ensure_ascii=False) + "\n")
    print(
        f"Wrote {OUTPUT_PATH}: {len(lexicon['courts'])} courts, "
        f"{len(lexicon['journals'])} journals, {len(lexicon['stopwords'])} stopwords, "
        f"{len(lexicon['synonyms'])} synonym groups, {len(lexicon['normalizations'])} "
        "normalizations"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script**

Run: `python scripts/extract_token_lexicon.py`
Expected: prints counts (courts/journals/stopwords/synonyms/normalizations all > 0), and
`packages/common/src/common/data/legal_lexicon.json` exists.

- [ ] **Step 3: Sanity-check the output**

Run: `python -c "import json; d = json.load(open('packages/common/src/common/data/legal_lexicon.json')); print('HIGHCOURT' in d['courts'] or 'HIGH COURT' in d['courts']); print(d['synonyms'].get('ACIT')); print(d['normalizations'].get('115I'))"`
Expected: prints `True`, `['ASSISTANT COMMISSIONER INCOME TAX', 'ACIT']`, `115-I` (confirms
the known sample rows from the design doc's audit made it through the classification correctly).

- [ ] **Step 4: Commit**

```bash
git add scripts/extract_token_lexicon.py packages/common/src/common/data/legal_lexicon.json
git commit -m "feat: extract centax-node token.js into typed JSON lexicon"
```

---

### Task 2: `legal_lexicon.py` — data + regex layer

**Files:**
- Create: `packages/common/src/common/legal_lexicon.py`
- Test: `packages/common/tests/test_legal_lexicon.py`

**Interfaces:**
- Consumes: `packages/common/src/common/data/legal_lexicon.json` (Task 1's output).
- Produces (used by Task 3):
  - `is_known_court(token: str) -> bool`
  - `is_known_journal(token: str) -> bool`
  - `is_stopword(token: str) -> bool`
  - `expand_synonyms(token: str) -> list[str]` — returns `[token]` if no synonym entry exists.
  - `normalize(token: str) -> str` — returns the token unchanged if no normalization exists.
  - `SECTION_PATTERN: re.Pattern` — matches `Section 54F`, `Sec. 12`, `u/s 80C`, `Rule 6(3)`,
    `Article 226` (case-insensitive).
  - `CITATION_PATTERN: re.Pattern` — matches `2024 ITR 123`, `(2023) 5 SCC 100`,
    `AIR 2022 SC 456` style citations.
  - `PARTY_PATTERN: re.Pattern` — matches `X vs Y` / `X v. Y` / `X versus Y`.

- [ ] **Step 1: Write the failing tests**

```python
import pytest
from common.legal_lexicon import (
    is_known_court, is_known_journal, is_stopword, expand_synonyms, normalize,
    SECTION_PATTERN, CITATION_PATTERN, PARTY_PATTERN,
)


def test_is_known_court_recognizes_high_court_and_bench_zonetypes():
    assert is_known_court("HIGHCOURT") is True
    assert is_known_court("AAR") is True  # BENCH-zoned in token.js
    assert is_known_court("NOTACOURT") is False


def test_is_known_journal_recognizes_itr():
    assert is_known_journal("ITR") is True
    assert is_known_journal("RANDOMWORD") is False


def test_is_stopword_recognizes_able():
    assert is_stopword("ABLE") is True
    assert is_stopword("SECTION") is False


def test_expand_synonyms_returns_alternatives_for_acit():
    result = expand_synonyms("ACIT")
    assert "ASSISTANT COMMISSIONER INCOME TAX" in result
    assert "ACIT" in result


def test_expand_synonyms_returns_input_unchanged_when_no_entry():
    assert expand_synonyms("RANDOMWORD") == ["RANDOMWORD"]


def test_normalize_applies_section_number_dash_fix():
    assert normalize("115I") == "115-I"


def test_normalize_returns_input_unchanged_when_no_entry():
    assert normalize("RANDOMWORD") == "RANDOMWORD"


@pytest.mark.parametrize("text", ["Section 54F", "Sec. 12", "u/s 80C", "Rule 6(3)", "Article 226"])
def test_section_pattern_matches_known_provision_formats(text):
    assert SECTION_PATTERN.search(text) is not None


def test_section_pattern_does_not_match_plain_number():
    assert SECTION_PATTERN.search("the number 80C alone") is None


@pytest.mark.parametrize("text", ["2024 ITR 123", "(2023) 5 SCC 100", "AIR 2022 SC 456"])
def test_citation_pattern_matches_known_citation_formats(text):
    assert CITATION_PATTERN.search(text) is not None


@pytest.mark.parametrize("text", ["Krishana Goel vs. Principal Commissioner", "State v. Doe", "X versus Y"])
def test_party_pattern_matches_vs_variants(text):
    assert PARTY_PATTERN.search(text) is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/common/tests/test_legal_lexicon.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'common.legal_lexicon'`

- [ ] **Step 3: Write the implementation**

```python
import json
import re
from pathlib import Path

_DATA_PATH = Path(__file__).parent / "data" / "legal_lexicon.json"
_LEXICON = json.loads(_DATA_PATH.read_text())

_COURTS = set(_LEXICON["courts"])
_JOURNALS = set(_LEXICON["journals"])
_STOPWORDS = set(_LEXICON["stopwords"])
_SYNONYMS = _LEXICON["synonyms"]
_NORMALIZATIONS = _LEXICON["normalizations"]


def is_known_court(token: str) -> bool:
    return token.upper() in _COURTS


def is_known_journal(token: str) -> bool:
    return token.upper() in _JOURNALS


def is_stopword(token: str) -> bool:
    return token.upper() in _STOPWORDS


def expand_synonyms(token: str) -> list[str]:
    return _SYNONYMS.get(token.upper(), [token])


def normalize(token: str) -> str:
    return _NORMALIZATIONS.get(token.upper(), token)


# Layer B: structural regex, hand-written (grammar, not data) - see design doc section 3.
SECTION_PATTERN = re.compile(
    r"\b(?:section|sec\.?|u/s|rule|article)\s*\d+[A-Za-z]*(?:\(\d+\))?", re.IGNORECASE
)
CITATION_PATTERN = re.compile(
    r"\(?\d{4}\)?\s*\(?\d*\)?\s*[A-Z]{2,10}\s*\d+", re.IGNORECASE
)
PARTY_PATTERN = re.compile(r"\bv(?:s\.?|ersus)?\.?\s+\w", re.IGNORECASE)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/common/tests/test_legal_lexicon.py -v`
Expected: all PASS. If `CITATION_PATTERN`/`SECTION_PATTERN`/`PARTY_PATTERN` regexes don't match
a given parametrized case, tighten/loosen that single regex until all parametrized cases pass -
these three patterns are the only implementation detail in this task open to adjustment.

- [ ] **Step 5: Commit**

```bash
git add packages/common/src/common/legal_lexicon.py packages/common/tests/test_legal_lexicon.py
git commit -m "feat: add legal_lexicon module (data + regex layers)"
```

---

### Task 3: `query_tokenizer.py` — procedural rules + query-shape classifier

**Files:**
- Create: `packages/common/src/common/query_tokenizer.py`
- Test: `packages/common/tests/test_query_tokenizer.py`

**Interfaces:**
- Consumes: `common.legal_lexicon` (Task 2): `is_known_court`, `is_known_journal`,
  `is_stopword`, `expand_synonyms`, `normalize`, `SECTION_PATTERN`, `CITATION_PATTERN`,
  `PARTY_PATTERN`.
- Produces (used by Task 4):
  - `classify_query_shape(query: str) -> str` — returns `"citation"`, `"provision"`, or
    `"plain"`.
  - `normalize_citation_spacing(query: str) -> str` — e.g. `"2024taxman.com"` ->
    `"2024 taxman.com"`.
  - `merge_keyword_number(tokens: list[str]) -> list[str]` — merges a keyword token with an
    immediately following number token (e.g. `["Section", "6"]` -> `["Section 6"]`);
    backtracks (leaves tokens unmerged) if the next token isn't numeric.
  - `merge_court_city(tokens: list[str]) -> list[str]` — merges a city name with an adjacent
    "High Court"/"Court" token (e.g. `["Delhi", "High", "Court"]` -> `["Delhi High Court"]`);
    backtracks if no court-type token follows.
  - `strip_stopwords(tokens: list[str]) -> list[str]` — drops tokens in the stopword set,
    but never drops a token that `is_known_journal` recognizes (journal-never-discard rule).
  - `extract_quoted_phrases(query: str) -> list[str]` — quoted substrings as single tokens,
    plus the remaining unquoted text split into individual word tokens.

- [ ] **Step 1: Write the failing tests**

```python
from common.query_tokenizer import (
    classify_query_shape, normalize_citation_spacing, merge_keyword_number,
    merge_court_city, strip_stopwords, extract_quoted_phrases,
)


def test_classify_query_shape_detects_citation():
    assert classify_query_shape("2024 ITR 123 exemption") == "citation"


def test_classify_query_shape_detects_party_citation():
    assert classify_query_shape("Krishana Goel vs. Principal Commissioner of Income-tax") == "citation"


def test_classify_query_shape_detects_provision():
    assert classify_query_shape("Section 54F exemption eligibility") == "provision"


def test_classify_query_shape_defaults_to_plain():
    assert classify_query_shape("can a company claim depreciation on goodwill") == "plain"


def test_classify_query_shape_prefers_citation_when_both_patterns_present():
    # a citation with a section number embedded is still primarily a citation lookup
    assert classify_query_shape("2024 ITR 123 on Section 54F") == "citation"


def test_normalize_citation_spacing_splits_year_from_source():
    assert normalize_citation_spacing("2024taxman.com 123") == "2024 taxman.com 123"


def test_normalize_citation_spacing_leaves_normal_text_unchanged():
    assert normalize_citation_spacing("Section 54F exemption") == "Section 54F exemption"


def test_merge_keyword_number_merges_section_and_number():
    assert merge_keyword_number(["Section", "6", "Income"]) == ["Section 6", "Income"]


def test_merge_keyword_number_backtracks_when_no_number_follows():
    assert merge_keyword_number(["Section", "Income"]) == ["Section", "Income"]


def test_merge_court_city_merges_delhi_high_court():
    assert merge_court_city(["Delhi", "High", "Court", "ruling"]) == ["Delhi High Court", "ruling"]


def test_merge_court_city_backtracks_when_no_court_token_follows():
    assert merge_court_city(["Delhi", "weather"]) == ["Delhi", "weather"]


def test_strip_stopwords_removes_recognized_stopword():
    assert "ABLE" not in strip_stopwords(["ABLE", "Section", "6"])


def test_strip_stopwords_never_drops_a_known_journal():
    # ITR is both plausible-stopword-adjacent and a real journal - must survive
    assert "ITR" in strip_stopwords(["citation", "in", "ITR"])


def test_extract_quoted_phrases_keeps_quoted_text_as_one_token():
    assert '"Income India"' not in extract_quoted_phrases('Section 6 of "Income India" in case of Supreme court')
    assert "Income India" in extract_quoted_phrases('Section 6 of "Income India" in case of Supreme court')
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/common/tests/test_query_tokenizer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'common.query_tokenizer'`

- [ ] **Step 3: Write the implementation**

```python
import re

from common.legal_lexicon import (
    CITATION_PATTERN, PARTY_PATTERN, SECTION_PATTERN, is_known_journal, is_stopword,
)

_CITATION_SPACING_PATTERN = re.compile(r"(\d{4})([a-zA-Z])")


def normalize_citation_spacing(query: str) -> str:
    """Ports queryAnalyzer.js's citation-format spacing step: '2024taxman.com' ->
    '2024 taxman.com'. A 4-digit year glued directly to a letter is always a citation-source
    boundary in this domain (no legitimate legal term starts with 4 digits then a letter)."""
    return _CITATION_SPACING_PATTERN.sub(r"\1 \2", query)


def classify_query_shape(query: str) -> str:
    """No-LLM query-shape classification for Instant mode boost selection. Citation checked
    first: a query naming both a citation and a section (e.g. "2024 ITR 123 on Section 54F")
    is still fundamentally a lookup for that one citation, so citation wins ties."""
    normalized = normalize_citation_spacing(query)
    if CITATION_PATTERN.search(normalized) or PARTY_PATTERN.search(normalized):
        return "citation"
    if SECTION_PATTERN.search(normalized):
        return "provision"
    return "plain"


def merge_keyword_number(tokens: list[str]) -> list[str]:
    """Ports queryAnalyzer.js's KEYWORD-type merge rule: a keyword followed by a number
    merges into one token ("Section" + "6" -> "Section 6"); if the next token isn't
    numeric, backtrack by leaving both tokens as they were."""
    result = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and tokens[i + 1].isdigit():
            result.append(f"{tokens[i]} {tokens[i + 1]}")
            i += 2
        else:
            result.append(tokens[i])
            i += 1
    return result


_COURT_TYPE_TOKENS = {"court", "high", "tribunal"}


def merge_court_city(tokens: list[str]) -> list[str]:
    """Ports queryAnalyzer.js's Court-type merge rule: a city/location name immediately
    followed by one or more court-type tokens ("High", "Court") merges into a single
    court-name token; backtracks (leaves tokens separate) if no court-type token follows."""
    result = []
    i = 0
    while i < len(tokens):
        span_end = i + 1
        while span_end < len(tokens) and tokens[span_end].lower() in _COURT_TYPE_TOKENS:
            span_end += 1
        if span_end > i + 1:
            result.append(" ".join(tokens[i:span_end]))
            i = span_end
        else:
            result.append(tokens[i])
            i += 1
    return result


def strip_stopwords(tokens: list[str]) -> list[str]:
    """Ports queryAnalyzer.js's stopword-skip rule, with the journal-never-discard
    exception: a token recognized as a journal abbreviation is kept even if it would
    otherwise look like a stopword."""
    return [t for t in tokens if is_known_journal(t) or not is_stopword(t)]


_QUOTED_PHRASE_PATTERN = re.compile(r'"([^"]+)"')


def extract_quoted_phrases(query: str) -> list[str]:
    """Ports queryAnalyzer.js's quoted-phrase extraction: a double-quoted substring becomes
    one token; the remaining unquoted text is split into individual word tokens."""
    phrases = _QUOTED_PHRASE_PATTERN.findall(query)
    remainder = _QUOTED_PHRASE_PATTERN.sub(" ", query)
    words = [w for w in remainder.split() if w]
    return phrases + words
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/common/tests/test_query_tokenizer.py -v`
Expected: all PASS. If `classify_query_shape`'s citation-vs-provision precedence test fails,
the fix is in `classify_query_shape`'s ordering (citation check must run before the section
check), not in the regexes from Task 2.

- [ ] **Step 5: Commit**

```bash
git add packages/common/src/common/query_tokenizer.py packages/common/tests/test_query_tokenizer.py
git commit -m "feat: add query_tokenizer module (procedural rules + shape classifier)"
```

---

### Task 4: Wire the fix into `es_client.raw_search`

**Files:**
- Modify: `packages/common/src/common/es_client.py:6-8,32-44`
- Modify: `packages/common/tests/test_es_client.py:59-92` (the two `raw_search` tests)

**Interfaces:**
- Consumes: `common.query_tokenizer.classify_query_shape` (Task 3).
- No change to `raw_search`'s own signature (`raw_search(client, query: str, limit: int = 20) -> list[dict]`) or return shape - existing callers (`instant/search.py`) are unaffected.

- [ ] **Step 1: Write the failing tests (replace the two existing `raw_search` tests)**

Replace `test_raw_search_returns_doc_id_score_heading_subheading` and
`test_raw_search_defaults_missing_heading_subheading_to_empty_string` in
`packages/common/tests/test_es_client.py` with:

```python
@pytest.mark.asyncio
async def test_raw_search_returns_doc_id_score_heading_subheading():
    client = FakeAsyncES(search_hits=[
        {
            "_source": {
                "id": "d1",
                "heading": "[2022] 140 taxmann.com 136 (Punjab & Haryana)",
                "subheading": "Krishana Goel vs. Principal Chief Commissioner of Income-tax",
            },
            "_score": 4.2,
        },
    ], index="researchindex_aic_test")

    results = await raw_search(client, "exemption claim", limit=20)

    assert results == [{
        "doc_id": "d1",
        "score": 4.2,
        "heading": "[2022] 140 taxmann.com 136 (Punjab & Haryana)",
        "subheading": "Krishana Goel vs. Principal Chief Commissioner of Income-tax",
    }]
    assert client.searched_index == "researchindex_aic_test"


@pytest.mark.asyncio
async def test_raw_search_defaults_missing_heading_subheading_to_empty_string():
    client = FakeAsyncES(search_hits=[{"_source": {"id": "d1"}, "_score": 1.0}])

    results = await raw_search(client, "query", limit=20)

    assert results == [{"doc_id": "d1", "score": 1.0, "heading": "", "subheading": ""}]


@pytest.mark.asyncio
async def test_raw_search_queries_heading_subheading_fullcontent_not_just_sparse_fields():
    client = FakeAsyncES(search_hits=[])

    await raw_search(client, "exemption claim", limit=20)

    query = client.search_calls[0]
    should_fields = {
        clause["multi_match"]["fields"][0] if "multi_match" in clause else None
        for clause in query["function_score"]["query"]["bool"]["should"]
    }
    for field in ("heading", "subheading", "fullcontent"):
        assert field in should_fields, f"{field} missing from should clauses: {should_fields}"


@pytest.mark.asyncio
async def test_raw_search_wraps_query_in_function_score_with_boost_fields():
    client = FakeAsyncES(search_hits=[])

    await raw_search(client, "exemption claim", limit=20)

    query = client.search_calls[0]
    assert "function_score" in query
    factor_fields = {f["field_value_factor"]["field"] for f in query["function_score"]["functions"]}
    assert factor_fields == {"documenttypeboost", "court_boost", "landmarkruling"}


@pytest.mark.asyncio
async def test_raw_search_excludes_blacklisted_landmarkruling_docs():
    client = FakeAsyncES(search_hits=[])

    await raw_search(client, "exemption claim", limit=20)

    query = client.search_calls[0]
    bool_clause = query["function_score"]["query"]["bool"]
    assert {"term": {"landmarkruling": -10}} in bool_clause.get("must_not", [])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/common/tests/test_es_client.py -v -k raw_search`
Expected: FAIL - the new tests fail because `raw_search` still builds the old flat
`multi_match` body; the two replaced tests fail because they now assert against the same
`raw_search` call but the surrounding file no longer matches (only if signatures changed -
here they should still pass structurally once query shape updates, confirm no accidental
break).

- [ ] **Step 3: Write the implementation**

```python
from elasticsearch import AsyncElasticsearch

from common.config import Settings
from common.query_tokenizer import classify_query_shape
from common.schemas import MASTERINFO_CITATION_FIELDS

_BOOST_PROFILES = {
    "citation": {"heading": 5.0, "subheading": 3.0, "fullcontent": 1.0,
                 "facts_text": 1.0, "held_text": 1.0, "headnotes_text": 1.5},
    "provision": {"heading": 2.0, "subheading": 3.0, "fullcontent": 1.0,
                  "facts_text": 1.0, "held_text": 1.0, "headnotes_text": 2.5},
    "plain": {"heading": 2.0, "subheading": 2.0, "fullcontent": 1.5,
              "facts_text": 1.0, "held_text": 1.0, "headnotes_text": 1.0},
}


class IndexedESClient:
    """Wraps a real ES client with the index name it should query, sourced
    from Settings.es_index (env-driven) at construction time - no index name
    is ever hardcoded downstream."""

    def __init__(self, client: AsyncElasticsearch, index: str):
        self._client = client
        self.index = index

    def __getattr__(self, name):
        return getattr(self._client, name)


def get_es_client(settings: Settings) -> IndexedESClient:
    auth = (settings.es_username, settings.es_password) if settings.es_username else None
    client = AsyncElasticsearch(
        settings.es_uri, basic_auth=auth, verify_certs=settings.es_verify_certs,
    )
    return IndexedESClient(client, settings.es_index)


def _build_field_query(query: str, shape: str) -> dict:
    """Query-shape-aware multi-field search (design doc section 1+3): every content field
    is searched (facts_text/held_text/headnotes_text are only 26-58% populated on the real
    index, so heading/subheading/fullcontent - 100% populated - must never be skipped),
    with boosts picked by the no-LLM query-shape classifier."""
    boosts = _BOOST_PROFILES[shape]
    return {
        "bool": {
            "should": [
                {"multi_match": {"query": query, "fields": [field], "boost": boost, "fuzziness": "AUTO"}}
                for field, boost in boosts.items()
            ],
            "minimum_should_match": 1,
        }
    }


def _wrap_function_score(field_query: dict) -> dict:
    """Ranking fix (design doc section 2): court_boost/documenttypeboost/landmarkruling are
    real, populated, precomputed boost fields the live index already carries but nothing in
    this codebase used before. documenttypeboost/landmarkruling constants are centax's own
    already-tuned formula for these exact fields; court_boost's factor is new, sized to that
    field's own smaller value range (0-294)."""
    return {
        "function_score": {
            "query": {
                "bool": {
                    "must": [field_query],
                    "must_not": [{"term": {"landmarkruling": -10}}],
                }
            },
            "functions": [
                {"field_value_factor": {"field": "documenttypeboost", "factor": 0.2, "modifier": "sqrt", "missing": 0.0001}},
                {"field_value_factor": {"field": "court_boost", "factor": 0.01, "modifier": "none", "missing": 0.0001}},
                {"field_value_factor": {"field": "landmarkruling", "factor": 1.2, "modifier": "log2p", "missing": 0.0001}},
            ],
            "boost_mode": "multiply",
        }
    }


async def raw_search(client, query: str, limit: int = 20) -> list[dict]:
    shape = classify_query_shape(query)
    body = _wrap_function_score(_build_field_query(query, shape))
    response = await client.search(index=client.index, query=body, size=limit)
    results = []
    for hit in response["hits"]["hits"]:
        source = hit["_source"]
        results.append({
            "doc_id": source["id"],
            "score": hit["_score"],
            "heading": source.get("heading", ""),
            "subheading": source.get("subheading", ""),
        })
    return results
```

Note: the `_should_fields` test in Step 1 reads
`clause["multi_match"]["fields"][0]` — each `should` clause here has exactly one field per
`multi_match` (per-field boosting, not one shared multi-field `multi_match`), so this matches.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/common/tests/test_es_client.py -v`
Expected: all PASS, including the untouched `resolve_doc_id_allowlist`/`fetch_*` tests below
in the same file (this task only touches `raw_search` and its two constants above it).

- [ ] **Step 5: Run the full test suite**

Run: `uv run pytest`
Expected: all tests pass (should be 226 + the ~24 new tests from Tasks 2-4).

- [ ] **Step 6: Commit**

```bash
git add packages/common/src/common/es_client.py packages/common/tests/test_es_client.py
git commit -m "fix: search always-populated fields and add real boost signals to Instant mode"
```

---

## Self-Review Notes

- Spec coverage: Section 1 (field fix) -> Task 4's `_build_field_query`. Section 2
  (`function_score`) -> Task 4's `_wrap_function_score`. Section 3 (query-shape boosting,
  all three layers) -> Tasks 1-3. Data extraction section -> Task 1. Testing section ->
  every task's Steps 1-4 plus Task 4 Step 5's full-suite run.
- The design doc's "before/after comparison against the live index" (Testing section) is
  informal/manual, matching this repo's existing `retrieval_eval.py` posture - not encoded
  as a plan task since it has no fixed pass/fail assertion; run manually after Task 4 if
  desired, not required for the plan to be complete.
- `_TERM_FILTER_FIELDS`/`resolve_doc_id_allowlist` (AI Mode's broken court/act/section/bench
  filters, confirmed 0% populated during the design audit) are untouched - out of scope
  per the design doc's Non-goals, tracked for a future AI Mode spec instead.
