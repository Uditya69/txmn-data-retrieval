import re

from common.legal_lexicon import (
    CITATION_PATTERN, PARTY_PATTERN, SECTION_PATTERN, expand_synonyms, is_known_journal, is_stopword,
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


def expand_query_synonyms(query: str) -> str:
    """Broadens recall by appending lexicon synonyms/abbreviation expansions for each token
    (e.g. "ACIT" -> also "ASSISTANT COMMISSIONER INCOME TAX") so multi_match's default OR
    semantics pick up documents phrased with the alternate term. Only appends new terms -
    never reorders or removes the original query - so classify_query_shape's regexes, which
    run on the un-expanded text before this, are unaffected."""
    tokens = query.split()
    seen = {t.upper() for t in tokens}
    extra = []
    for token in tokens:
        for synonym in expand_synonyms(token):
            if synonym.upper() not in seen:
                seen.add(synonym.upper())
                extra.append(synonym)
    return f"{query} {' '.join(extra)}" if extra else query


_SECTION_KEYWORDS = {"section", "sec", "sec.", "u/s", "rule", "article"}
_SECTION_NUMBER_PATTERN = re.compile(r"^\d+[A-Za-z]*(\(\w+\))*$")


def merge_keyword_number(tokens: list[str]) -> list[str]:
    """Ports queryAnalyzer.js's KEYWORD-type merge rule: a keyword (section/sec/u/s/rule/
    article - the same set SECTION_PATTERN recognizes) followed by a section-number-shaped
    token merges into one token ("Section" + "6" -> "Section 6", "Section" + "5(8)" ->
    "Section 5(8)"); if tokens[i] isn't a recognized keyword or the next token isn't
    section-number-shaped, backtrack by leaving both tokens as they were. Without the
    keyword check, this used to merge ANY word followed by a bare number ("Spa" + "175" ->
    "Spa 175"), producing boost phrases that don't exist verbatim in any document and
    shredding real citations like "175 taxmann.com 251" into unrelated fragments; without
    the letter/subsection allowance, genuine refs like "Section 5(8)" or "Section 69C"
    never merged at all since "5(8)"/"69C" aren't pure digits."""
    result = []
    i = 0
    while i < len(tokens):
        if (
            i + 1 < len(tokens)
            and tokens[i].lower() in _SECTION_KEYWORDS
            and _SECTION_NUMBER_PATTERN.match(tokens[i + 1])
        ):
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


def merge_citation_span(tokens: list[str]) -> list[str]:
    """Merges a <volume-number> <journal-abbreviation> <page-number> run into a single
    phrase token ("133" + "taxmann.com" + "196" -> "133 taxmann.com 196") - the same
    general citation shape CITATION_PATTERN recognizes, but driven by the data-backed
    is_known_journal lexicon rather than a regex, since real query text writes the
    journal abbreviation with punctuation ("taxmann.com") that a bare letter-run regex
    wouldn't reliably bound. Without this, a citation's three tokens stayed loose and
    unboosted - the only phrase boosts available were Section/Rule/Article refs, so a
    query naming an exact case citation had no way to boost the one span most likely to
    pin the correct document. Backtracks (leaves tokens separate) unless all three
    tokens are present and shaped correctly."""
    result = []
    i = 0
    while i < len(tokens):
        if (
            i + 2 < len(tokens)
            and tokens[i].isdigit()
            and is_known_journal(tokens[i + 1])
            and tokens[i + 2].isdigit()
        ):
            result.append(" ".join(tokens[i:i + 3]))
            i += 3
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


def extract_boost_phrases(query: str) -> list[str]:
    """Runs the full queryAnalyzer.js-ported pipeline (quote grouping -> citation-span merge
    -> keyword+number merge -> court+city merge -> stopword strip) and returns only the
    multi-word survivors - a quoted span or a merge like "133 taxmann.com 196"/"Section 6"/
    "Delhi High Court" is a precise concept worth a phrase-match boost layered on top of the
    loose base query, unlike a lone leftover word. Citation-span merge runs first since it's
    the most specific (three-token) shape - keyword+number merge only looks at two tokens at
    a time, so running it first could never consume a mid-span token a citation merge needs."""
    tokens = extract_quoted_phrases(query)
    tokens = merge_citation_span(tokens)
    tokens = merge_keyword_number(tokens)
    tokens = merge_court_city(tokens)
    tokens = strip_stopwords(tokens)
    return [t for t in tokens if " " in t]
