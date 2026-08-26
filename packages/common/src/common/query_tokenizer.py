import re

from common.legal_lexicon import (
    CITATION_PATTERN, KNOWN_ACT_NAMES, PARTY_PATTERN, SECTION_PATTERN, expand_synonyms,
    is_known_court, is_known_journal, is_stopword, normalize,
)

_CITATION_SPACING_PATTERN = re.compile(r"(\d{4})([a-zA-Z])")


def normalize_citation_spacing(query: str) -> str:
    """Ports queryAnalyzer.js's citation-format spacing step: '2024taxman.com' ->
    '2024 taxman.com'. A 4-digit year glued directly to a letter is always a citation-source
    boundary in this domain (no legitimate legal term starts with 4 digits then a letter)."""
    return _CITATION_SPACING_PATTERN.sub(r"\1 \2", query)


# Allowlist, not a denylist - SECTION_PATTERN/CITATION_PATTERN/PARTY_PATTERN (legal_lexicon.py)
# and the merge rules above only ever produce/expect word chars, whitespace, and this fixed set
# of legal-typography punctuation: . , - / ( ) " ' & : ; % (citation spacing "133 taxmann.com
# 196", section subrefs "5(8)(a)", party names "Tata & Sons"/"O'Brien", quoted phrases, "u/s").
# Any character outside that set never carries legal meaning here and, left glued to an
# adjacent token (e.g. an accidental trailing "\" from a copy-paste), can defeat
# merge_keyword_number's exact-shape match (_SECTION_NUMBER_PATTERN) - "55\" silently fails to
# merge into a "section 55" anchor chunk, so classify_intent_mode falls through to "hybrid" and
# the ES query body's own match_phrase boosts never fire, even though the user typed a precise
# section lookup. An allowlist covers every such stray character (not just ones already seen
# breaking something), rather than growing a denylist one incident at a time. Stripped once,
# upstream of every consumer (chunk_query, the ES multi_match/match_phrase text, embed calls),
# so none of them can see a different query text.
_ALLOWED_QUERY_CHARS_PATTERN = re.compile(r"[^\w\s.,\-/()\"'&:;%]", re.UNICODE)


def strip_noise_characters(query: str) -> str:
    """Removes any character outside this domain's legal-typography allowlist (see comment
    above) - anything that can't legitimately appear in a citation, section/rule reference, Act
    name, party name, or court/journal token. Collapses any whitespace left behind by the strip
    so it stays idempotent and doesn't introduce new stopword-shaped gaps."""
    stripped = _ALLOWED_QUERY_CHARS_PATTERN.sub(" ", query)
    return re.sub(r"\s+", " ", stripped).strip()


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


# Targets already wired at a more specific point than generic query-text broadening -
# expanding the raw query with the literal word "CASELAWS"/"RULE"/"ARTICLE" would inject a
# group-name token that never appears in real document body text (see detect_group_signals'
# should-clause boost, below), and SEC's "SECTION" target already only fires when adjacent to
# a section number (see merge_keyword_number). Excluded here so expand_query_normalizations
# doesn't double up on what those two already handle.
_NORMALIZATION_EXPANSION_EXCLUDED_TARGETS = {"CASELAWS", "RULE", "ARTICLE", "SECTION"}


def expand_query_normalizations(query: str) -> str:
    """Broadens recall the same way expand_query_synonyms does, but from legal_lexicon's
    separate `normalizations` dict (1299 entries, zero key overlap with `synonyms` -
    confirmed) - plain abbreviation/department-name entries like ASST -> ASSISTANT,
    DBODCIRCULAR -> DBOD, CHALLANNO -> CHALLAN that were sitting completely unwired
    (normalize() was called nowhere in the real query-building pipeline before
    detect_group_signals/merge_keyword_number). Restricted to single-word targets -
    normalizations' ~1200 multi-word Act-name-expansion entries
    (RESTRICTIVETRADEPRACTICESACT -> "RESTRICTIVE TRADE PRACTICES ACT") are deliberately left
    out: unverified whether real queries ever contain the glued-together acronym form at all,
    a much bigger/riskier lift than this. Only appends new terms - never reorders or removes
    the original query, same contract as expand_query_synonyms."""
    tokens = query.split()
    seen = {t.upper() for t in tokens}
    extra = []
    for token in tokens:
        target = normalize(token)
        if target == token or " " in target or target in _NORMALIZATION_EXPANSION_EXCLUDED_TARGETS:
            continue
        if target.upper() not in seen:
            seen.add(target.upper())
            extra.append(target)
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
    clean adjacency shapes.

    The keyword itself also runs through legal_lexicon's normalize() ("sec" -> "SECTION",
    per legal_lexicon.json's normalizations dict) before joining - without this, "sec 55"
    produced the merged phrase text "sec 55" verbatim, which never match_phrase-hits a real
    doc heading written "Section 55" even though chunk-type detection already correctly
    recognized "sec" as a section keyword for routing purposes. "u/s"/"sec." have no
    normalizations entry (nothing verified to canonicalize them to) and pass through
    unchanged - only wire what the lexicon actually backs, don't guess an expansion."""
    result = []
    i = 0
    while i < len(tokens):
        if (
            i + 1 < len(tokens)
            and tokens[i].lower() in _SECTION_KEYWORDS
            and _SECTION_NUMBER_PATTERN.match(tokens[i + 1])
        ):
            result.append(f"{normalize(tokens[i])} {tokens[i + 1]}")
            i += 2
        elif (
            i + 1 < len(tokens)
            and tokens[i + 1].lower() in _SECTION_KEYWORDS
            and _SECTION_NUMBER_PATTERN.match(tokens[i])
        ):
            result.append(f"{normalize(tokens[i + 1])} {tokens[i]}")
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


# Chunk types chunk_query() already recognizes as a precise, lookup-safe anchor - a doc
# either contains this exact span or it doesn't, so ES lexical search alone resolves it as
# well as Milvus dense/sparse would (see classify_intent_mode).
_ANCHOR_CHUNK_TYPES = {"section", "citation", "court_city", "quoted"}


def _is_bare_act_name(text: str) -> bool:
    """True when `text` is nothing but a known Act name, optionally trailed by a bare
    year ("Income Tax Act 1961") - chunk_query has no dedicated Act-name chunk type, so a
    query like "what is the income tax act" leaves it in the generic text-run chunk; this
    re-checks that leftover text against KNOWN_ACT_NAMES rather than teaching chunk_query
    a fifth merge rule for a check only classify_intent_mode needs. Anything beyond the
    act name itself (a trailing concept word, e.g. "income tax act depreciation") fails
    both checks below and correctly falls through to hybrid - see
    test_classify_intent_mode_tags_conceptual_query_as_hybrid."""
    lowered = text.lower()
    if lowered in KNOWN_ACT_NAMES:
        return True
    for act_name in KNOWN_ACT_NAMES:
        if lowered.startswith(act_name) and lowered[len(act_name):].strip().isdigit():
            return True
    return False


def classify_intent_mode(query: str) -> str:
    """Tags a query "keyword" (ES-only search is sufficient) or "hybrid" (needs the full
    Milvus dense/sparse + RRF pipeline). "keyword" fires only when EVERY surviving chunk
    (after chunk_query's own filler/stopword strip) is a precise, unambiguous anchor - a
    structural chunk type (section/rule/article number, citation, court/city, quoted
    phrase) or a bare known court/journal token/Act name. Deliberately excludes
    legal_lexicon's `synonyms` entries (PE, ALP, AMT, ...): those are abbreviations for
    broad legal concepts, not lookup keys, and still need semantic (Milvus) recall - see
    test_classify_intent_mode_does_not_tag_synonym_lexicon_term_as_keyword. Any leftover
    unanchored content (a bare concept word, a party name, a mix of anchor + concept text)
    tags hybrid, same as an empty/all-filler query."""
    chunks = chunk_query(query)
    if not chunks:
        return "hybrid"
    for chunk in chunks:
        if chunk["type"] in _ANCHOR_CHUNK_TYPES:
            continue
        text = chunk["text"]
        if " " not in text and (is_known_court(text) or is_known_journal(text)):
            continue
        if _is_bare_act_name(text):
            continue
        return "hybrid"
    return "keyword"


# A legal_lexicon normalization target isn't always the real ES `groups.group.name` value it
# signals: RULING/CASE/CITATION/JUDGEMENT do normalize to "CASELAWS", which is also the real
# group name, but ARTICLE's real group is "Experts Opinion" (verified against a real doc
# pulled into the 2026-08-25 investigation: heading "[2019] 106 taxmann.com 47 (Article)",
# groups.group.name == "Experts Opinion", id 111050000000000051) - centax-node's own token
# table (constants/token.js) confirms the same three group ids/boosts (CASELAWS
# 111050000000000060/10,000,000; RULE 111050000000000026/2,000,000; ARTICLE's entry points at
# 111050000000000051/2,000,000, i.e. Experts Opinion). CIRCULAR/NOTIFICATION have their own
# confirmed token-table entries too, but are deliberately excluded here - we don't have
# verified live data for those two groups' behavior on this repo's own index (see es_client.py
# comment for CASELAWS/RULE/ARTICLE's own verification caveat, which still applies even to
# these three).
_GROUP_SIGNAL_ES_GROUP_NAMES = {
    "CASELAWS": "CASELAWS",
    "RULE": "RULE",
    "ARTICLE": "Experts Opinion",
}


def detect_group_signals(chunks: list[dict]) -> set[str]:
    """Real ES `groups.group.name` values signaled by any word across chunk_query()'s chunks -
    ported from centax-node's queryAnalyzer.js/token dictionary recognizing certain words
    (RULING, JUDGEMENT, CASE, RULE, ARTICLE, ...) as meaning the query wants that content type
    specifically, not just matching them as ordinary text. A signal word is often not its own
    chunk: "landmark Supreme Court ruling on GST" leaves "ruling" merged into the trailing
    "ruling GST" text-run chunk (see chunk_query), so this checks every word inside each
    chunk's text, not each chunk's text as a whole.

    legal_lexicon's normalize() covers suffixed/abbreviated variants (RULES/RULENO -> RULE,
    CASE/CASES/CITATION/JUDGEMENT -> CASELAWS) but has no identity entry for a bare word
    already in its own canonical form (RULE -> RULE, ARTICLE -> ARTICLE would be a no-op
    normalization, so the lexicon data omits it) - normalize()'s own fallback (return the
    input unchanged when not found) makes checking the *normalized* form still catch this,
    since a bare "RULE" normalizes to "RULE" right back.

    Callers turn each returned group name into a should-clause `groups.group.name.keyword`
    boost sized to compete with _PHRASE_BOOSTS (see es_client.py) - without it, a query like
    the one above scores generic "Words & Idioms" commentary docs whose heading literally
    contains "Supreme Court" far above the real case law it's asking for, since
    heading/subheading/headnotes_text phrase-match boosts fire identically regardless of
    document type."""
    return {
        _GROUP_SIGNAL_ES_GROUP_NAMES[target]
        for chunk in chunks
        for word in chunk["text"].split()
        for target in [normalize(word.upper())]
        if target in _GROUP_SIGNAL_ES_GROUP_NAMES
    }


def build_dense_sparse_query(chunks: list[dict], fallback: str) -> str:
    """Reconstructs a cleaned search string for Milvus dense embedding + native sparse search
    from chunk_query's own chunks - drops the conversational/question-word scaffolding
    chunk_query already strips via strip_stopwords ("what is section 55" -> chunks == [{"text":
    "section 55", ...}]) while keeping every real content word, citation, and section
    reference, in order. Exists because Instant mode has no LLM query-rewrite step (unlike AI
    Mode's extract_intent) to do this - without it, "section 55" and "what is section 55" send
    identical text to ES's phrase-boost clauses (chunk_query already cleans those) but
    different, noise-diluted text to Milvus dense/sparse, which search the raw sentence
    verbatim. Falls back to the original query on the empty-chunks edge case (empty query)."""
    text = " ".join(chunk["text"] for chunk in chunks if chunk.get("text"))
    return text or fallback
