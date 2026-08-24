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
    """Generic no-LLM citation/provision/plain query-shape classifier. No longer drives
    Instant mode's ES boost-profile selection (see common.instant_classifier.effective_label
    and es_client.py) or backend routing - retired from that role only. Still a legitimate
    generic utility with a real caller: retrieval_api.ai_mode.intent's anchor-detection floor
    (_has_legal_anchor/build_lexicon_check), an unrelated AI Mode concern. Citation checked
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
# Normalizes EVERY separator variant between a section-type keyword and its number into a
# single space, BEFORE tokenization ever runs: "Section52" (fully glued, zero separator),
# "Section-52" (dash, zero space), "Section - 52" (spaced dash), "Section- 52"/"Section -52"
# (asymmetric spacing), en/em dash - all become "Section 52". This corpus's own `heading` field
# is literally formatted "Section - 52" (see es_client.py's _PHRASE_BOOSTS comment) -
# a user copy-pasting a heading, a citation, or just typing fast hits one of these shapes
# constantly, and .split() handles none of them consistently on its own (a glued token like
# "Section52"/"Section-52" never gets split at all; a spaced dash becomes its own token that
# breaks the plain two-token adjacency check merge_keyword_number does below). Both the
# whitespace (\s*) and the dash ([-–—]?) are optional and independent of each other, so this one
# pattern covers all five shapes above with no separate branch per shape - including the
# zero-separator case (\s* and the dash class both match empty, leaving only the (?=\d)
# lookahead to require a digit immediately follows). It's also idempotent on an
# already-clean "Section 52": \s* consumes the existing single space, the substitution puts
# an equivalent single space back. Normalizing once, upstream, at the string level - rather
# than teaching every downstream token-matching branch to also tolerate a separator - means
# merge_keyword_number only ever needs to handle the single already-clean "keyword number"
# shape, same as it always did. (?=\d) keeps this scoped to what precedes an actual
# section-number-shaped token, so it never touches an unrelated word that happens to start
# with one of these keywords but isn't followed by a number (e.g. "sectional").
_SECTION_KEYWORD_SEPARATOR_PATTERN = re.compile(
    r"(?i)\b(" + "|".join(re.escape(k) for k in _SECTION_KEYWORDS) + r")\s*[-–—]?\s*(?=\d)"
)


def normalize_section_dash(query: str) -> str:
    """'Section52' / 'Section-52' / 'Section - 52' / 'Section- 52' / 'Section -52' ->
    'Section 52' - every separator shape (glued, dash, spaced dash, asymmetric spacing)
    collapses to one space before tokenization, so they all produce the same merged chunk and
    hit the same match_phrase boost as the plain space-separated form."""
    return _SECTION_KEYWORD_SEPARATOR_PATTERN.sub(r"\1 ", query)


def merge_keyword_number(tokens: list[str]) -> list[str]:
    """Ports queryAnalyzer.js's KEYWORD-type merge rule: a keyword (section/sec/u/s/rule/
    article - the same set SECTION_PATTERN recognizes) adjacent to a section-number-shaped
    token merges into one token, canonicalized to "keyword number" order regardless of which
    order the user typed them in ("Section" + "6" -> "Section 6", "6" + "Section" -> "Section
    6", "Section" + "5(8)" -> "Section 5(8)"); if neither adjacent pair matches, backtrack by
    leaving both tokens as they were. Without the keyword check, this used to merge ANY word
    followed by a bare number ("Spa" + "175" -> "Spa 175"), producing boost phrases that don't
    exist verbatim in any document and shredding real citations like "175 taxmann.com 251"
    into unrelated fragments; without the letter/subsection allowance, genuine refs like
    "Section 5(8)" or "Section 69C" never merged at all since "5(8)"/"69C" aren't pure digits.
    The reversed "number keyword" order (e.g. "55 section") must canonicalize to "keyword
    number" too, not just merge in place - downstream chunk-type detection only recognizes a
    merged phrase as a section reference when the keyword is its first word. Assumes
    normalize_section_dash already ran (see extract_quoted_phrases) - by the time tokens reach
    here, every dash variant is already collapsed to a plain space, so this only needs the two
    clean adjacency shapes."""
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
        elif (
            i + 1 < len(tokens)
            and tokens[i + 1].lower() in _SECTION_KEYWORDS
            and _SECTION_NUMBER_PATTERN.match(tokens[i])
        ):
            result.append(f"{tokens[i + 1]} {tokens[i]}")
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
    one token; the remaining unquoted text is split into individual word tokens. Runs
    normalize_section_dash() first (entry point shared by chunk_query and extract_boost_phrases)
    so a glued "Section-52" is already "Section 52" - two tokens - by the time .split() runs,
    the same as it would be for a quoted span containing that shape."""
    query = normalize_section_dash(query)
    phrases = _QUOTED_PHRASE_PATTERN.findall(query)
    remainder = _QUOTED_PHRASE_PATTERN.sub(" ", query)
    words = [w for w in remainder.split() if w]
    return phrases + words


# analyze_query() used to live here as a UI-trace-only diagnostic that independently
# re-derived shape/expansion/phrases stage-by-stage, separate from what the real search path
# (es_client.raw_search) computed. It was removed after that independence caused a real bug:
# raw_search moved to chunk_query() (proximity-phrase chunking - groups an unrecognized word
# run like "Dimension Data India" into one phrase), but analyze_query kept using its own older
# pipeline that never did that grouping, so the UI trace showed a stale breakdown that didn't
# match what ES actually received. Use chunk_query() directly, or es_client.build_query_preview()
# for the full shape+chunks+ES-query breakdown - both single-sourced from what raw_search itself
# calls, so this can't drift out of sync again.


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


# Ported from centax-node's queryAnalyzer.js ProximityDefault class (services/queryAnalyzer.js) -
# each becomes an ES match_phrase `slop` value (see searchTextElastic.js, which sends
# `{"match_phrase": {field: {query, slop: element.Proximity}}}` per token). 0 = exact phrase
# (ProximityDefault.Phrase/DateType), 2 = citation triples tolerate minor gaps
# (ProximityDefault.Citation), 5 = an ordinary run of words (ProximityDefault.Text/DefaultValue).
_PROXIMITY_EXACT = 0
_PROXIMITY_CITATION = 2
_PROXIMITY_TEXT = 5

_ZERO_PAD_WIDTH = 3


def _zero_padded_variant(number_part: str) -> str | None:
    """Legacy generates a zero-padded numeric alternative for a short section/rule number
    (production getLowLevelQuery output for "section 92C" resolved to
    "SECTION 92C | SECTION 092C" - some older citations format the numeric part zero-padded).
    Only the numeric prefix is padded; a letter/subsection suffix ("C", "(1)(c)") is kept as-is.
    Returns None when there's nothing to pad (already >= 3 digits, or no leading digits)."""
    match = re.match(r"^(\d+)(.*)$", number_part)
    if not match:
        return None
    digits, suffix = match.groups()
    if len(digits) >= _ZERO_PAD_WIDTH:
        return None
    return f"{digits.zfill(_ZERO_PAD_WIDTH)}{suffix}"


def _classify_merged_chunk(token: str) -> tuple[str, int, str | None]:
    """A merged (multi-word) token from the extract_boost_phrases pipeline doesn't carry which
    merge rule produced it, so this re-derives that from the token's own shape - cheaper than
    threading provenance through every merge function, and the shapes don't overlap (a
    Section-keyword-led token can't also look like a citation triple or a court-city pair)."""
    words = token.split()
    first_word = words[0].rstrip(".").lower()
    if first_word in _SECTION_KEYWORDS and len(words) >= 2:
        alt_number = _zero_padded_variant(words[-1])
        alt_text = f"{' '.join(words[:-1])} {alt_number}" if alt_number else None
        return "section", _PROXIMITY_EXACT, alt_text
    if len(words) == 3 and words[0].isdigit() and words[2].isdigit() and is_known_journal(words[1]):
        return "citation", _PROXIMITY_CITATION, None
    if any(w.lower() in _COURT_TYPE_TOKENS for w in words[1:]):
        return "court_city", _PROXIMITY_EXACT, None
    return "quoted", _PROXIMITY_EXACT, None


def chunk_query(query: str) -> list[dict]:
    """Groups a query into ES match_phrase-ready chunks, closing a real gap versus the older
    extract_boost_phrases: centax-node's queryAnalyzer.js converts the ENTIRE query into
    phrase-with-slop matches, not just the few explicitly-recognized shapes (Section+number,
    court+city, citation triple, quotes) - anything else still gets grouped. Its default/
    fallback rule (queryAnalyzer.js:465-491, TokenType.Text) accumulates any maximal run of
    consecutive non-keyword words into one contiguous phrase token with slop=5
    (ProximityDefault.Text), split only at recognized keyword boundaries. That's what makes a
    party name like "Dimension Data India" - no Section/Court/citation keyword anywhere near
    it - one precise phrase in centax-node's own getLowLevelQuery output, instead of three
    independent OR'd terms the way extract_boost_phrases (and multi_match on the raw query)
    leaves it: a doc merely containing "Dimension" OR "Data" OR "India" anywhere, in any order,
    scores the same as one where they appear together as the actual party name.
    Does not decide field boosts - callers apply whatever per-field weight they choose per
    chunk; this only decides how the query text is grouped and how much positional slop each
    group tolerates when matched."""
    tokens = extract_quoted_phrases(query)
    tokens = merge_citation_span(tokens)
    tokens = merge_keyword_number(tokens)
    tokens = merge_court_city(tokens)
    tokens = strip_stopwords(tokens)

    chunks: list[dict] = []
    text_run: list[str] = []

    def flush_text_run():
        if text_run:
            chunks.append({"text": " ".join(text_run), "proximity": _PROXIMITY_TEXT, "type": "text", "alt_text": None})
            text_run.clear()

    for token in tokens:
        if " " in token:
            flush_text_run()
            chunk_type, proximity, alt_text = _classify_merged_chunk(token)
            chunks.append({"text": token, "proximity": proximity, "type": chunk_type, "alt_text": alt_text})
        else:
            text_run.append(token)
    flush_text_run()
    return chunks
