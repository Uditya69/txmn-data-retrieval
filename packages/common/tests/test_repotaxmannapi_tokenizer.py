"""Tests for common.repotaxmannapi_tokenizer.tokenize() - a Python port of the token-scanning
loop in repotaxmannapi/TaxmannAPI/Elastic/TaxmannQueryAnalizer.cs (a separate, read-only .NET
codebase; see that file for the ported source). Every test below was derived by hand-tracing a
concrete query string through the real C# source (line citations are on the corresponding
implementation code in repotaxmannapi_tokenizer.py, not repeated here) - not guessed."""
import datetime

from common.repotaxmannapi_tokenizer import RepotaxmannapiToken, tokenize


def test_stop_word_alone_produces_no_tokens():
    """'THE' is dictionary element_type "99" (StopWord). ProcessorQuery's StopWord case
    (TaxmannQueryAnalizer.cs:1699-1700) does `break` with no token pushed, and since TempToken
    was never given a Type it is never pushed at the loop's end either (line 1956)."""
    assert tokenize("the") == []


def test_keyword_or_stop_word_with_no_next_word_is_silently_dropped():
    """'AO' is element_type "65" (KeyWordOrStopWord, TaxmannQueryAnalizer.cs:1667-1677).
    ProcessKeyWord("AO", ...) sees GetElement() return "" (end of list) so its own
    `if (NextWord.Trim().Length > 0)` guard (line 1068) is never entered and it returns false.
    Unlike the plain KeyWord case (1602-1625), the KeyWordOrStopWord case has no `else` branch,
    so the word is dropped entirely - not even pushed as literal text."""
    assert tokenize("AO") == []


def test_plain_unrecognized_words_accumulate_into_one_text_token():
    """Neither "hello" nor "world" has a dictionary entry, so CheckForKeyword always fails for
    them (TaxmannQueryAnalizer.cs:567-598), and every other detector (numeric/date/month/
    citation/etc.) also fails, so both fall into the default text-accumulation branch
    (1911-1949), which appends into a single TokenType.Text ("TX") token."""
    tokens = tokenize("hello world")
    assert tokens == [
        RepotaxmannapiToken(
            query_text="hello world", query_date=None, org_text="", type="TX",
            or_in=False, proximity=5, group_id="0",
        )
    ]


def test_lone_number_produces_number_token():
    """"999" is purely numeric so `isNo` is true (line 1815), but ProcessCitation("Y",...),
    ProcessCitation("V",...), ProcessDate, and IsDayMonthFormat all fail immediately because
    GetElement() returns "" (end of list) for every lookahead they attempt. Falling through to
    line 1868-1873 creates a TokenType.Number ("N") token with ProximityDefault.Numeric (6)."""
    tokens = tokenize("999")
    assert tokens == [
        RepotaxmannapiToken(
            query_text="999", query_date=None, org_text="", type="N",
            or_in=False, proximity=6, group_id="0",
        )
    ]


def test_citation_pattern_year_volume_journal_page():
    """"223 ITR 1" drives ProcessCitation's "Y"->"V"->"J"->"P" state chain
    (TaxmannQueryAnalizer.cs:1320-1501): "223" is numeric so entry point "Y" advances to "V"
    with NextWord="ITR"; ITR is a Journal (element_type "88") so "V" advances to "J" with
    NextWord="1"; "1" is numeric so "J" advances to "P"; "P" hits end-of-list and finalizes
    Citation="223 ITR 1". GetKeyProximity("CITATION") is 0 in the extracted dictionary."""
    tokens = tokenize("223 ITR 1")
    assert tokens == [
        RepotaxmannapiToken(
            query_text="223 ITR 1", query_date=None, org_text="", type="CT",
            or_in=False, proximity=0, group_id="111050000000000060",
        )
    ]


def test_slash_date_is_parsed_as_date_format():
    """"01/02/2020" has no internal spaces so it survives the top-level `Split(' ')`
    (TaxmannQueryAnalizer.cs:168) as a single QueryElement; WordStripSplChr then turns each
    "/" into a space (regex \\W -> " ", line 1533-1538), producing the single element
    "01 02 2020" (WITH internal spaces). IsDateFormat (737-762) turns spaces back into "/" and
    parses it with day-first (en-GB) semantics via `Split('/').Length == 3`, giving 1 Feb 2020.
    The token is created via the 4-arg CreateToken(BufferObj, DateFmt, ProximityDefault.
    DateType=1, false) at line 1894, which formats QueryText as "dd MM yyyy" (398)."""
    tokens = tokenize("01/02/2020")
    assert tokens == [
        RepotaxmannapiToken(
            query_text="01 02 2020", query_date=datetime.date(2020, 2, 1), org_text="",
            type="DT", or_in=False, proximity=1, group_id="0",
        )
    ]


def test_month_and_year_is_parsed_as_month_format():
    """"Jan" is not in the dictionary so CheckForKeyword fails and control falls to
    IsMonthFormat (847-913): MonthList().ContainsKey("JAN") is true, so it looks ahead one more
    element ("2020"), sees it's numeric and longer than 3 chars, and parses "01 Jan 2020".
    CreateToken's 3-arg overload (418-440) with tType==MonthFmt formats QueryText as
    "MMM yyyy" (line 402/430), giving "Jan 2020"."""
    tokens = tokenize("Jan 2020")
    assert tokens == [
        RepotaxmannapiToken(
            query_text="Jan 2020", query_date=datetime.date(2020, 1, 1), org_text="",
            type="MT", or_in=False, proximity=1, group_id="0",
        )
    ]


def test_keyword_alone_pushes_key_search_text_as_keyword_token():
    """"APDIR" is element_type "66" (KeyWord). ProcessKeyWord("APDIR", ...) sees GetElement()
    return "" (end of list) so it returns false (TaxmannQueryAnalizer.cs:1057-1068); since
    "APDIR" != "AS" the KeyWord case's else-branch (1612-1624) pushes GetKeySearchText("APDIR")
    ("A P DIR") as a plain KeyWord ("KW") token."""
    tokens = tokenize("APDIR")
    assert tokens == [
        RepotaxmannapiToken(
            query_text="A P DIR", query_date=None, org_text="", type="KW",
            or_in=False, proximity=5, group_id="111050000000000057",
        )
    ]


def test_keyword_followed_by_number_builds_leading_zero_alternation():
    """"SECTION 10": SECTION is element_type "66" (KeyWord). ProcessKeyWord("SECTION", ...)
    (1057-1190): NextWord="10", not a dictionary word, first digit '1' != '0' and
    GetKeySearchText("SECTION") ("SECTION") has no "|" and no "ICDS |", so it falls to the
    final `else` (1154-1180) building "SECTION 10 | SECTION 010" (leading-zero alternative,
    NextWord2 stays empty since "10" contains none of I/i/O/o/P/p). The KeyWord case
    (1602-1625) pushes this as a SectionTypeFormat ("T1") token."""
    tokens = tokenize("SECTION 10")
    assert tokens == [
        RepotaxmannapiToken(
            query_text="SECTION 10 | SECTION 010", query_date=None, org_text="", type="T1",
            or_in=False, proximity=0, group_id="111050000000000064",
        )
    ]


def test_keyword_with_leading_roman_numeral_next_word():
    """"SECTION IV": SECTION is element_type "66" (KeyWord), search_text "SECTION" (no "|").
    ProcessKeyWord("SECTION", ...) (1057-1190): NextWord="IV" is not itself a dictionary word
    (GetResource("IV") is empty), so the first `if` (1070-1073) is skipped. NextWord's first
    char 'I' fails int.TryParse (isNum1 stays false), and
    ConstantsRoman.ConvertRomanNumber("IV") = 4 succeeds as isNum, so the `else if` at
    1074-1103 fires: isNum=4>0, GetResource("SECTION").IndexOf("|") is -1 (no "|" in
    "SECTION"), so SearchText = GetKeySearchText("SECTION") + " " + "IV" = "SECTION IV"
    (1095), functionReturnValue=true. This never reaches the digit-leading NextWord branch
    (1104-1186) at all - it is the "leading roman numeral, non-digit-start" branch. The
    KeyWord case (1602-1625) pushes the result as a SectionTypeFormat ("T1") token with
    GetKeyProximity("SECTION")=0 and SECTION's group_id."""
    tokens = tokenize("SECTION IV")
    assert tokens == [
        RepotaxmannapiToken(
            query_text="SECTION IV", query_date=None, org_text="", type="T1",
            or_in=False, proximity=0, group_id="111050000000000064",
        )
    ]


def test_keyword_digit_leading_next_word_with_p_triggers_swap_alternation():
    """"SECTION 80P": NextWord="80P" is not a dictionary word (unlike e.g. "194I"/"10i", which
    ARE dictionary NumAlphaZone entries and would instead be back-tracked and rescanned
    separately - deliberately avoided here to isolate this branch). NextWord's first char '8'
    parses as a digit, so the digit-leading branch (1104-1186) is entered. NextWord2 is built
    by TaxmannQueryAnalizer.cs:1112-1123's ladder of Contains/Replace checks: "80P" contains no
    " I"/"I"/" i"/"i"/" O"/"O"/" o"/"o"/" P", but does contain plain "P", so
    `NextWord2 = NextWord.Replace("P", " P")` = "80 P" (line 1121). NextWord's first char is
    not "0" (skips 1125-1133's leading-zero branch), GetKeySearchText("SECTION") ("SECTION")
    contains no "ICDS |" (skips 1134-1153), and GetResource("SECTION") contains no "|" so the
    plain `else` at 1171-1174 fires: SearchText = "SECTION 80P | SECTION 080P". Since
    NextWord2.Length>0, 1175-1178 appends " | SECTION 80 P | SECTION 080 P". Final:
    "SECTION 80P | SECTION 080P | SECTION 80 P | SECTION 080 P", pushed as a
    SectionTypeFormat ("T1") token via the KeyWord case (1602-1625)."""
    tokens = tokenize("SECTION 80P")
    assert tokens == [
        RepotaxmannapiToken(
            query_text="SECTION 80P | SECTION 080P | SECTION 80 P | SECTION 080 P",
            query_date=None, org_text="", type="T1",
            or_in=False, proximity=0, group_id="111050000000000064",
        )
    ]


def test_icds_keyword_with_numeric_next_word_converts_to_roman():
    """"ICDS 5": ICDS is element_type "66" (KeyWord) with search_text
    "ICDS | INCOME COMPUTATION AND DISCLOSURE STANDARDS" (a real dictionary entry, unlike the
    unreachable KeyWordOnly/Synonym/Month element types). ProcessKeyWord("ICDS", ...):
    NextWord="5" is not a dictionary word, its first char '5' parses as a digit
    (isNo2=true, line 1108-1109), so control enters the digit-leading branch. NextWord2 stays
    empty ("5" contains none of I/i/O/o/P/p). NextWord's first char is not "0" (skips
    1125-1133). GetKeySearchText("ICDS").IndexOf("ICDS |")>=0 is true AND
    int.TryParse("5", out isNumber3) succeeds, so the ICDS special case fires (1134-1153):
    newNextword = ConstantsRoman.NumericToRoman(5) = "V". GetResource("ICDS").IndexOf("|")>=0
    is true (search_text itself contains "|"), so SearchText is rebuilt by splitting
    "ICDS | INCOME COMPUTATION AND DISCLOSURE STANDARDS" on '|' and appending " V" to each
    trimmed part, joined back with '|': "ICDS V|INCOME COMPUTATION AND DISCLOSURE STANDARDS V"
    (1139-1150), functionReturnValue=true. Pushed as a SectionTypeFormat ("T1") token via the
    KeyWord case (1602-1625) with GetKeyProximity("ICDS")=2 and ICDS's group_id."""
    tokens = tokenize("ICDS 5")
    assert tokens == [
        RepotaxmannapiToken(
            query_text="ICDS V|INCOME COMPUTATION AND DISCLOSURE STANDARDS V",
            query_date=None, org_text="", type="T1",
            or_in=False, proximity=2, group_id="111050000000011660",
        )
    ]


def test_keyword_type2_with_no_next_word_uses_semicolon_split_search_text():
    """"CBDTCIRCULAR" alone is element_type "64" (KeyWordType2). ProcessKeyWordType2
    (1191-1238): NextWord is "" (end of list), so the outer `else` (1229-1235) splits
    GetKeySearchText("CBDTCIRCULAR") on ';' into StElement[0]/[1] but leaves
    functionReturnValue false (never set true on this path) - it returns false even though
    SearchText/OthText were assigned by reference. The KeyWordType2 case's false-branch
    (1637-1646) then pushes GetKeySearchText(RText) UNSPLIT as a KeyWord token, followed by
    OthText (StElement[1], the by-ref value from the failed call) as a second KeyWord token
    with proximity 1 (hardcoded at line 1634/1643)."""
    tokens = tokenize("CBDTCIRCULAR")
    whole = (
        "circular;cbdt circular | Central Board of Direct Taxes circular | "
        "DIRECT TAX CIRCULAR | INCOME TAX CIRCULAR | WEALTH TAX CIRCULAR"
    )
    oth = (
        "cbdt circular | Central Board of Direct Taxes circular | "
        "DIRECT TAX CIRCULAR | INCOME TAX CIRCULAR | WEALTH TAX CIRCULAR"
    )
    assert tokens == [
        RepotaxmannapiToken(
            query_text=whole, query_date=None, org_text="", type="KW",
            or_in=False, proximity=2, group_id="111050000000000057",
        ),
        RepotaxmannapiToken(
            query_text=oth, query_date=None, org_text="", type="KW",
            or_in=False, proximity=1, group_id="111050000000000057",
        ),
    ]


def test_high_court_keyword_followed_by_city_name():
    """"HIGHCOURT DELHI": HIGHCOURT is element_type "67" (KeyWordHighCourt).
    ProcessKeyWordHighCourt (1239-1289): NextWord="DELHI" is not a stop word, and
    GetResource("DELHI") is non-empty with element_type "89" (Court), so it builds
    "HIGH COURT DELHI" from GetKeySearchText("HIGHCOURT").Split('|') (only one item: "HIGH
    COURT") plus the city name. Pushed as SectionTypeFormat ("T1")."""
    tokens = tokenize("HIGHCOURT DELHI")
    assert tokens == [
        RepotaxmannapiToken(
            query_text="HIGH COURT DELHI", query_date=None, org_text="", type="T1",
            or_in=False, proximity=2, group_id="0",
        )
    ]


def test_high_court_lookup_is_case_sensitive_bug_preserved():
    """Real bug found by reading, not guessed: ProcessKeyWordHighCourt's dictionary check at
    TaxmannQueryAnalizer.cs:1252 is `GetResource(NextWord.Replace(" ", "")).ToUpper()` - the
    `.ToUpper()` is applied to GetResource's RETURN VALUE, not to NextWord before the lookup
    (unlike the correctly-normalized check in IsNextWordHighCourt at line 929). Since GetResource
    itself does no case folding, a lowercase next word like "delhi" misses the dictionary (keys
    are uppercase), so ProcessKeyWordHighCourt fails and backtracks "delhi" back onto the
    cursor untouched. The KeyWordHighCourt case's `else` branch (1658-1666) - taken on ANY
    failure, not just this bug - still pushes GetKeySearchText("HIGHCOURT") ("HIGH COURT") as
    its own KeyWord token. "delhi" is then rescanned on the next loop iteration: CheckForKeyword
    DOES correctly uppercase (`sMatchText.ToUpper()`, line 587), so it matches the "DELHI"
    dictionary entry (Court, "89") after all, lands in the `default:` switch arm, IsNextWordHighCourt
    fails (end of list), and it falls through to plain text - text using GetKeySearchText("delhi")
    ("DELHI", from the dictionary, uppercase) rather than the user's original lowercase spelling."""
    tokens = tokenize("HIGHCOURT delhi")
    assert tokens == [
        RepotaxmannapiToken(
            query_text="HIGH COURT", query_date=None, org_text="", type="KW",
            or_in=False, proximity=2, group_id="0",
        ),
        RepotaxmannapiToken(
            query_text="DELHI", query_date=None, org_text="", type="TX",
            or_in=False, proximity=5, group_id="0",
        ),
    ]


def test_zone_keyword_pushes_search_zone_token_directly():
    """"GOI" is element_type "77" (Zone). The Zone case (1678-1684) pushes
    GetKeySearchText("GOI") directly as a SearchZone ("Z") token with no Process* call
    involved at all."""
    tokens = tokenize("GOI")
    assert tokens == [
        RepotaxmannapiToken(
            query_text="GOVERNMENT INDIA | GOI", query_date=None, org_text="", type="Z",
            or_in=False, proximity=2, group_id="0",
        )
    ]


def test_num_alpha_zone_keyword_pushes_num_alpha_zone_token():
    """"105I" is element_type "87" (NumAlphaZone). The NumAlphaZone case (1685-1691) pushes
    GetKeySearchText("105I") ("105-I") as an "NZ" token."""
    tokens = tokenize("105I")
    assert tokens == [
        RepotaxmannapiToken(
            query_text="105-I", query_date=None, org_text="", type="NZ",
            or_in=False, proximity=0, group_id="0",
        )
    ]


def test_country_keyword_pushes_keyword_token():
    """"AFGHANISTAN" is element_type "56" (Country). The Country case (1692-1698) pushes
    GetKeySearchText("AFGHANISTAN") as a plain KeyWord ("KW") token (the IsCountry flag it also
    sets only affects the isGlobalSearch/ARTICLE post-processing at 1968-1984, which is out of
    scope for this query)."""
    tokens = tokenize("AFGHANISTAN")
    assert tokens == [
        RepotaxmannapiToken(
            query_text="AFGHANISTAN", query_date=None, org_text="", type="KW",
            or_in=False, proximity=2, group_id="0",
        )
    ]


def test_dated_keyword_followed_by_date_pushes_date_token_only():
    """"DATED 01/02/2020": DATED is element_type "69" (KeyWordDated). ProcessKeyWordDated
    (1290-1319) reads NextWord="01 02 2020" (see the slash-to-space note on the DT test above),
    IsDateFormat succeeds, and the KeyWordDated case (1707-1715) pushes only the resulting
    DateFmt token - the literal word "DATED" itself is never pushed as its own token on
    either the success or failure path (no `else` branch exists for this case)."""
    tokens = tokenize("DATED 01/02/2020")
    assert tokens == [
        RepotaxmannapiToken(
            query_text="01 02 2020", query_date=datetime.date(2020, 2, 1), org_text="",
            type="DT", or_in=False, proximity=1, group_id="0",
        )
    ]


def test_court_word_not_followed_by_high_court_falls_back_to_text():
    """"DELHI" alone is element_type "89" (Court), handled by the `default:` switch arm
    (1721-1805). IsNextWordHighCourt fails (end of list), so there's no `break`, and execution
    falls through into the generic text-accumulation code beneath the Court/Journal checks
    (1772-1804), which appends GetKeySearchText("DELHI") ("DELHI") into a Text ("TX") token -
    distinct from a totally-unrecognized word only in that it is dictionary-known but still
    ends up as text here."""
    tokens = tokenize("DELHI")
    assert tokens == [
        RepotaxmannapiToken(
            query_text="DELHI", query_date=None, org_text="", type="TX",
            or_in=False, proximity=5, group_id="0",
        )
    ]


def test_keyword_glued_to_digits_without_space():
    """"RULE10" (no space) never matches CheckForKeyword as a whole. It falls to
    CheckKeyWordWithOutSpace (702-735), which scans for the first digit run, splits into
    keyWrd="RULE" / paramStr="10", confirms GetResource("RULE") is element_type "66" (KeyWord),
    and since paramStr doesn't start with '0' builds "RULE 10" directly (no leading-zero
    alternation - that only happens inside ProcessKeyWord, which this helper does not call).
    Pushed as SectionTypeFormat ("T1") via ProcessorQuery's CheckKeyWordWithOutSpace branch
    (1903-1910)."""
    tokens = tokenize("RULE10")
    assert tokens == [
        RepotaxmannapiToken(
            query_text="RULE 10", query_date=None, org_text="", type="T1",
            or_in=False, proximity=2, group_id="111050000000000026",
        )
    ]


def test_slash_separated_month_slash_year_without_month_name():
    """"12/2020": WordStripSplChr turns the "/" into a space, giving the single QueryElement
    "12 2020". IsMonthFormat's second path (TaxmannQueryAnalizer.cs:885-906) explicitly guards
    for exactly this shape - `dtfr.Length == 2 && (dtfr[0].Length == 4 || dtfr[1].Length == 4)`
    - i.e. a 2-part "mm/yyyy" (or "yyyy/mm") where one part is a 4-digit year, and parses it
    with day defaulted to 1. (Initially mis-traced this as reaching isCirNoFormat/
    NotificationNumber instead - the explicit length-4 guard in the source makes clear this
    "N/YYYY" shape is IsMonthFormat's deliberate target, not an accident; corrected after
    re-reading, not guessed.)"""
    tokens = tokenize("12/2020")
    assert tokens == [
        RepotaxmannapiToken(
            query_text="Dec 2020", query_date=datetime.date(2020, 12, 1), org_text="",
            type="MT", or_in=False, proximity=1, group_id="0",
        )
    ]


def test_notification_style_number_when_no_part_is_four_digits():
    """"5.202": WordStripSplChr turns "." into a space, giving the single QueryElement
    "5 202". This does NOT satisfy IsMonthFormat's 4-digit-part guard (parts "5"/"202" are
    1 and 3 chars), nor IsDateFormat's 3-part requirement, nor any CheckKeyWordWithOutSpace
    hit, so it falls to isCirNoFormat's regex `(\\s+\\d+\\s|\\d+\\s|\\s+\\d)`, which matches
    "5 " (635-644), producing a NotificationNumber ("NN") token, proximity 3 (1911-1920)."""
    tokens = tokenize("5.202")
    assert tokens == [
        RepotaxmannapiToken(
            query_text="5 202", query_date=None, org_text="", type="NN",
            or_in=False, proximity=3, group_id="0",
        )
    ]


def test_ordinal_number_word_is_rewritten_to_words():
    """"21st" matches IsNumberInWordsType's regex `(\\d)+(st|nd|rd|th)` (645-700): iNo=21,
    20<21<30 so SearchText starts "twenty", iNo becomes 1, and iNo==1 appends " first" ->
    "twenty first". ProcessorQuery substitutes QryWrd with this before the generic text
    accumulation (1923), producing a Text ("TX") token with the rewritten text."""
    tokens = tokenize("21st")
    assert tokens == [
        RepotaxmannapiToken(
            query_text="twenty first", query_date=None, org_text="", type="TX",
            or_in=False, proximity=5, group_id="0",
        )
    ]


def test_quoted_phrase_produces_trailing_phrase_word_token():
    """A quoted phrase is extracted into PhraseElements entirely in the constructor
    (TaxmannQueryAnalizer.cs:127-159), before ProcessorQuery ever runs, and is appended to
    QueryTokens as PhraseWord ("PH") tokens only at the very end (1987-1993) - AFTER every
    token produced by the main scanning loop, regardless of where the quote appeared in the
    original query text. "tax" (outside the quotes) has no dictionary entry so it becomes a
    plain Text token first."""
    tokens = tokenize('"annual report" tax')
    assert tokens == [
        RepotaxmannapiToken(
            query_text="tax", query_date=None, org_text="", type="TX",
            or_in=False, proximity=5, group_id="0",
        ),
        RepotaxmannapiToken(
            query_text="annual report", query_date=None, org_text="", type="PH",
            or_in=False, proximity=0, group_id="0",
        ),
    ]


def test_multi_word_stop_word_is_removed_before_scanning():
    """"INCASEOF" (element_type "98", MultiWordStopWord) is the one entry of this type in the
    extracted dictionary. RemoveMultiWordStopWords (599-634) is invoked from the constructor
    in a `while` loop (line 185) and blanks out every QueryElement in the matched span (here
    "in"/"case"/"of") before ProcessorQuery ever sees them, leaving only "X" to be tokenized."""
    tokens = tokenize("in case of X")
    assert tokens == [
        RepotaxmannapiToken(
            query_text="X", query_date=None, org_text="", type="TX",
            or_in=False, proximity=5, group_id="0",
        )
    ]


def test_empty_query_produces_no_tokens():
    """EndOfList is true immediately (QueryElement.Length == 0 after splitting/trimming an
    empty string), so ProcessorQuery's main `while (!EndOfList)` loop body never runs
    (TaxmannQueryAnalizer.cs:1577) and there are no phrase elements to append either."""
    assert tokenize("") == []


def test_every_token_carries_a_group_id_field():
    """Not from the C# Token struct (which has no GroupID field at all - see the org_text
    finding below) but a controller-mandated addition for this repo's Task 10 (group-based
    scoring): every RepotaxmannapiToken must expose some group_id, defaulting to "0" for
    tokens not produced from a classified dictionary keyword (see module docstring for the
    full ruling)."""
    tokens = tokenize("hello")
    assert all(hasattr(t, "group_id") for t in tokens)
    assert tokens[0].group_id == "0"


def test_org_text_is_always_empty_matching_the_never_assigned_c_sharp_field():
    """Verified by reading the ENTIRE file: NewToken() (379-389) initializes OrgText to the
    C# default for an unassigned string field (effectively empty), and no code path anywhere
    in TaxmannQueryAnalizer.cs (searched exhaustively) ever assigns Token.OrgText. It is a
    vestigial field in the original struct. This port keeps the field (per the brief's
    interface spec) but it is always "" since there is nothing faithful to port for it."""
    for query in ("hello", "APDIR", "223 ITR 1", "GOI"):
        for token in tokenize(query):
            assert token.org_text == ""
