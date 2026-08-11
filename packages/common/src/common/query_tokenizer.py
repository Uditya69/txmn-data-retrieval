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
