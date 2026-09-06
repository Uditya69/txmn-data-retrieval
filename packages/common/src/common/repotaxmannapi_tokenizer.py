"""Python port of repotaxmannapi/TaxmannAPI/Elastic/TaxmannQueryAnalizer.cs (a separate,
read-only .NET codebase) - see docs/superpowers/specs/2026-09-01-repotaxmannapi-exact-replica-design.md
for why this exists as a parallel, opt-in path rather than replacing this repo's existing
(eval-verified, sum-mode) tokenizer.

Every method below is ported from a specific line range of that file, cited in its own
docstring/comments. The port was done by reading the entire ~2008-line file (not summaries of
it) before writing any of this code - see
.superpowers/sdd/2026-09-01-repotaxmannapi-exact-replica/task-3-report.md for the full record
of what was read and what each branch does.

Two deliberate, documented departures from a byte-for-byte port:

1. **group_id is per-token here, a single mutable "primary tag" in the C# original.**
   TaxmannQueryAnalizer.cs never puts a group id on its `Token` struct at all (the struct has
   no such field - see the org_text note below for the sibling finding). Instead, `iGroupID`
   is one field on the analyzer object, set by `SetPrimaryTag` (TaxmannQueryAnalizer.cs:288-317)
   the FIRST time any keyword is classified in a whole query, and left alone after that
   (`if (iTagNo == "0")` at line 294) - so in the original, group id is a query-level fact,
   not a token-level one. This repo's plan (Task 10, group-based scoring) needs group id
   addressable per token, so - per explicit controller ruling on this task - every
   RepotaxmannapiToken here carries its OWN group_id, taken from the dictionary entry for the
   keyword that produced it (not gated by "first classification wins"), defaulting to "0" for
   tokens with no originating keyword (plain text/number/date/citation/notification tokens).
   This is a deliberate deviation, not an oversight.

2. **org_text is always "".** Read the entire file looking for every assignment to
   `Token.OrgText` (the struct field at TaxmannQueryAnalizer.cs:88): there is none, anywhere.
   `NewToken()` (379-389) never sets it either, so it just holds a C# string field's default.
   It is vestigial in this file. Kept here only because the brief's interface spec (mirroring
   the `Token` struct) names it; it is always "".
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from common.repotaxmannapi_tokenizer_dates import (
    get_date_format,
    is_date_format,
    is_day_month_format,
    is_month_format,
    month_names,
    process_date,
)
from common.repotaxmannapi_token_dictionary import TokenDictEntry, load_repotaxmannapi_token_dictionary


def classify_token(word: str) -> TokenDictEntry | None:
    """Direct port of TaxmannQueryAnalizer.cs's GetResource(word.ToUpper()) lookup
    (TaxmannQueryAnalizer.cs:217-233) - the C# returns "" for a missing key, which
    SetPrimaryTag (lines 296-314) treats as "no dictionary entry"; this returns None for
    the same case so callers use a Pythonic None-check instead of an empty-string check."""
    return load_repotaxmannapi_token_dictionary().get(word.upper())


# ---------------------------------------------------------------------------------------------
# TaxmannQueryAnalizer.cs:58-92 - ElementType / TokenType / ProximityDefault / Token
# ---------------------------------------------------------------------------------------------
class ElementType:
    MONTH = "40"
    SYNONYM = "44"
    KEY_WORD_ONLY = "55"
    COUNTRY = "56"
    KEY_WORD_TYPE2 = "64"
    KEY_WORD_OR_STOP_WORD = "65"
    KEY_WORD = "66"
    KEY_WORD_HIGH_COURT = "67"
    KEY_WORD_DATED = "69"
    ZONE = "77"
    NUM_ALPHA_ZONE = "87"
    JOURNAL = "88"
    COURT = "89"
    SBTM = "90"
    MULTI_WORD_STOP_WORD = "98"
    STOP_WORD = "99"


class TokenType:
    PHRASE_WORD = "PH"
    KEY_WORD = "KW"
    SECTION_TYPE_FORMAT = "T1"
    SEARCH_ZONE = "Z"
    NUM_ALPHA_ZONE = "NZ"
    NUMBER = "N"
    NOTIFICATION_NUMBER = "NN"
    TEXT = "TX"
    CITATION = "CT"
    DATE_FMT = "DT"
    MONTH_FMT = "MT"
    DATE_MONTH_FMT = "DM"
    SYNONYM = "SY"


class ProximityDefault:
    PHRASE = 0
    DATE_TYPE = 1
    CITATION = 2
    NOTIFICATION = 3
    NUMERIC = 6
    TEXT = 5
    DEFAULT_VALUE = 5


# Hardcoded in LoadSynonymWordList (TaxmannQueryAnalizer.cs:100-108). NOTE: that method is
# never called anywhere in the file (verified by exhaustive search), so SynonymWordList is
# always empty at runtime in the real C# too - the Synonym branch below is genuinely dead code
# with the extracted dictionary (it has zero element_type "44" entries). Kept for structural
# fidelity; see task-3-report.md for the "unreachable, untested" note.
SYNONYM_WORD_LIST: dict[str, str] = {}


@dataclass
class RepotaxmannapiToken:
    """Mirrors the C# `Token` struct (TaxmannQueryAnalizer.cs:84-92), plus a `group_id` field
    not present on the original struct - see the module docstring for why."""

    query_text: str
    query_date: date | None
    org_text: str
    type: str
    or_in: bool
    proximity: int
    group_id: str = "0"


# ---------------------------------------------------------------------------------------------
# TaxmannQueryAnalizer.cs:194-233 - constructor helpers
# ---------------------------------------------------------------------------------------------
def _check_and_remove(search_text: str, remove_from_char: str) -> str:
    """Port of CheckAndRemove (TaxmannQueryAnalizer.cs:194-216): strips every occurrence of
    `remove_from_char` together with the rest of that "word" up to (not including) the next
    space. Used in the constructor to strip stray apostrophes."""
    ret_text = search_text
    while remove_from_char in ret_text:
        i_st = ret_text.index(remove_from_char)
        rest = ret_text[i_st:]
        if " " in rest:
            i_ed = ret_text.index(" ", i_st)
            ret_text = ret_text[:i_st] + ret_text[i_ed:]
        else:
            ret_text = ret_text[:i_st]
    return ret_text


def word_strip_spl_chr(text: str) -> str:
    """Direct port of WordStripSplChr (TaxmannQueryAnalizer.cs:1513-1539): if the string
    contains no "%", replace every run matched by `\\W` (regex non-word-char, i.e. anything
    that isn't [A-Za-z0-9_]) with a single space, collapse doubled spaces once, and trim.
    A string containing "%" is passed through untouched (`Str.IndexOf("%") == -1` guard)."""
    if "%" not in text:
        text = re.sub(r"\W", " ", text)
        text = text.replace("  ", " ")
    return text.strip()


def _get_word_dict_entry(word: str) -> TokenDictEntry | None:
    return classify_token(word)


def _get_key_search_text(key_text: str) -> str:
    """Port of GetKeySearchText (TaxmannQueryAnalizer.cs:360-376): "" when the dictionary
    entry has no search_text component (element has only 2 ':'-delimited parts in the raw
    resource string, i.e. `search_text is None` in the extracted table)."""
    entry = _get_word_dict_entry(key_text)
    if entry is None or entry["search_text"] is None:
        return ""
    return entry["search_text"].strip()


def _get_key_proximity(key_text: str) -> int:
    """Port of GetKeyProximity (TaxmannQueryAnalizer.cs:345-359)."""
    entry = _get_word_dict_entry(key_text)
    assert entry is not None
    return entry["proximity"]


def _is_stop_word(check_word: str) -> bool:
    """Port of IsStopWord (TaxmannQueryAnalizer.cs:549-566)."""
    entry = _get_word_dict_entry(check_word.replace(" ", ""))
    return entry is not None and entry["element_type"] == ElementType.STOP_WORD


def _is_cir_no_format(text: str) -> bool:
    """Port of isCirNoFormat (TaxmannQueryAnalizer.cs:635-644)."""
    return re.search(r"(\s+\d+\s|\d+\s|\s+\d)", text) is not None


def _is_number_in_words_type(elm_str: str) -> tuple[bool, str]:
    """Port of IsNumberInWordsType (TaxmannQueryAnalizer.cs:645-700): rewrites an ordinal
    numeral like "21st" into its word form "twenty first". Returns (matched, search_text)."""
    temp = elm_str.strip().lower()
    if re.search(r"(\d)+(st|nd|rd|th)", temp) is None:
        return False, ""

    i_no = int(temp[:-2])
    search_text = ""
    if 20 < i_no < 30:
        search_text = "twenty"
        i_no -= 20
    elif 30 < i_no < 40:
        search_text = "thirty"
        i_no -= 30
    elif 40 < i_no < 50:
        search_text = "forty"
        i_no -= 40
    elif 50 < i_no < 60:
        search_text = "fifty"
        i_no -= 50
    elif 60 < i_no < 70:
        search_text = "sixty"
        i_no -= 60
    elif 70 < i_no < 80:
        search_text = "seventy"
        i_no -= 70
    elif 80 < i_no < 90:
        search_text = "eighty"
        i_no -= 80
    elif 90 < i_no < 100:
        search_text = "ninty"
        i_no -= 90

    ones = {
        1: "first", 2: "second", 3: "third", 4: "forth", 5: "fifth", 6: "sixth",
        7: "seventh", 8: "eighth", 9: "ninth",
    }
    tens_words = {
        10: "tenth", 11: "eleventh", 12: "twelfth", 13: "thirteenth", 14: "fourteenth",
        15: "fifteenth", 16: "sixteenth", 17: "seventeenth", 18: "eighteenth",
        19: "nineteenth", 20: "twentieth", 30: "thirtieth", 40: "fortieth",
        50: "fiftieth", 60: "sixtieth", 70: "seventieth", 80: "eightieth",
        90: "ninetieth", 100: "hundredth",
    }
    if i_no in ones:
        search_text = (search_text + " " + ones[i_no]).strip() if search_text else ones[i_no]
    elif i_no in tens_words:
        search_text = tens_words[i_no]
    else:
        search_text = ""

    return True, search_text.strip()


def _remove_leading_zero(text: str) -> str:
    """Port of RemoveLeadingZero (TaxmannQueryAnalizer.cs:1503-1512)."""
    while len(text) > 1 and text[0] == "0":
        text = text[1:]
    return text


def _numeric_to_roman(num: int) -> str:
    """Port of ConstantsRoman.NumericToRoman (repotaxmannapi/TaxmannAPI/BL/Constants.cs:644+),
    used by ProcessKeyWord's ICDS-roman-numeral special case."""
    thou = ["", "M", "MM", "MMM"]
    hund = ["", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"]
    tens = ["", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"]
    ones = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"]
    if num >= 4000:
        hi, lo = divmod(num, 1000)
        return f"({_numeric_to_roman(hi)}){_numeric_to_roman(lo)}"
    result = thou[num // 1000]
    num %= 1000
    result += hund[num // 100]
    num %= 100
    result += tens[num // 10]
    num %= 10
    result += ones[num]
    return result


_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def _convert_roman_number(roman: str) -> int:
    """Port of ConstantsRoman.ConvertRomanNumber (Constants.cs:588+), simplified: the
    original's "(...)  " thousands-parenthesis extension is not exercised by anything this
    file's ProcessKeyWord passes it (ordinary section/rule numerals), so only the plain
    subtractive-notation loop is ported."""
    roman = roman.upper()
    if not roman:
        return 0
    total = 0
    last_value = 0
    try:
        for ch in reversed(roman):
            new_value = _ROMAN_VALUES[ch]
            if new_value < last_value:
                total -= new_value
            else:
                total += new_value
                last_value = new_value
    except KeyError:
        return 0
    return total


class _Analyzer:
    """Stateful port of the TaxmannQueryAnalizer class. Kept as an internal class (mirroring
    the C#'s imperative, cursor-based design almost 1:1) because the algorithm is inherently
    stateful - GetElement()/BackTrack() advance and rewind a shared cursor across many
    mutually-recursive helper methods, exactly as in the original."""

    def __init__(self, search_text: str, is_global: bool = False) -> None:
        # TaxmannQueryAnalizer.cs:114-193 (constructor)
        self.is_global = is_global
        # TaxmannQueryAnalizer.cs:1553-1554 - IsArticle/IsCountry locals inside
        # ProcessorQuery. This port constructs a fresh _Analyzer once per tokenize() call
        # and calls process_query() exactly once on it, so per-instance attributes are the
        # correct equivalent scope.
        self.is_article = False
        self.is_country = False
        text = search_text.strip()
        # Line 126: collapse "A - B" (dash surrounded by alnum, with optional spaces) to "A B".
        text = re.sub(r"(?<=[A-Za-z0-9])\s*-\s*(?=[A-Za-z0-9])", " ", text)
        self.phrase_elements: list[str] = []
        # Lines 129-159: extract "quoted phrases" into phrase_elements, removing them from text.
        text = text.replace('""', " ").strip()
        while ' "' in text or text.startswith('"'):
            i_st_len = 2
            i_st = text.find(' "')
            if i_st < 0:
                if text.startswith('"'):
                    i_st = 0
                    i_st_len = 1
                else:
                    break
            i_ed = text.find('" ', i_st + i_st_len)
            if i_ed < 0:
                if text.find('"', i_st + i_st_len) == len(text) - 1:
                    phrase_text = text[i_st + i_st_len : len(text) - 1]
                    self.phrase_elements.append(word_strip_spl_chr(phrase_text))
                    text = text[:i_st]
                else:
                    text = text[:i_st] + " " + text[i_st + 2 :]
            else:
                phrase_text = text[i_st + i_st_len : i_ed]
                self.phrase_elements.append(word_strip_spl_chr(phrase_text))
                text = text[:i_st] + " " + text[i_ed + 2 :]

        # Lines 161-186: split on space, strip apostrophes/special-chars per element, then
        # remove multi-word stop words.
        while "  " in text:
            text = text.replace("  ", " ")
        self.query_element: list[str] = text.split(" ")
        if len(self.query_element) > 0 and self.query_element != [""]:
            for i in range(len(self.query_element)):
                self.query_element[i] = _check_and_remove(self.query_element[i], "'")
                self.query_element[i] = word_strip_spl_chr(self.query_element[i])
            while self._remove_multi_word_stop_words():
                pass
        else:
            self.query_element = []

        self.i_ptr = 0

    # -------------------------------------------------------------------------------------
    # TaxmannQueryAnalizer.cs:244-286 - cursor primitives
    # -------------------------------------------------------------------------------------
    @property
    def end_of_list(self) -> bool:
        return self.i_ptr >= len(self.query_element)

    def get_element(self) -> str:
        while not self.end_of_list:
            element = self.query_element[self.i_ptr].strip()
            self.i_ptr += 1
            if len(element) > 0:
                return element
        return ""

    def back_track(self) -> None:
        self.i_ptr -= 1

    # -------------------------------------------------------------------------------------
    # TaxmannQueryAnalizer.cs:567-634 - keyword scanning / multi-word stop word removal
    # -------------------------------------------------------------------------------------
    def _check_for_keyword(self, start_index: int) -> tuple[bool, str, int]:
        """Port of CheckForKeyword (567-598): longest match, scanning from the end of the
        element array backward, concatenating (space-free, stop-words-excluded) candidate
        spans starting at start_index."""
        for j in range(len(self.query_element) - 1, start_index - 1, -1):
            s_match_text = ""
            for i in range(start_index, j + 1):
                if not _is_stop_word(self.query_element[i]):
                    s_match_text += self.query_element[i]
            match_text = ""
            if classify_token(s_match_text.replace(" ", "")) is not None:
                match_text = s_match_text.replace(" ", "")
            elif classify_token(s_match_text) is not None:
                match_text = s_match_text
            if len(match_text) > 0:
                return True, match_text, j
        return False, "", -1

    def _remove_multi_word_stop_words(self) -> bool:
        """Port of RemoveMultiWordStopWords (599-634): finds the first (earliest start,
        longest span) run of elements whose concatenation is a MultiWordStopWord ("98") entry,
        blanks every element in that span, and returns True; returns False when none remain."""
        for i_st in range(len(self.query_element)):
            for j in range(len(self.query_element) - 1, i_st - 1, -1):
                s_check_text = "".join(self.query_element[i_st : j + 1])
                s_check_text = s_check_text.replace(" ", "").upper()
                if not s_check_text:
                    continue
                entry = classify_token(s_check_text)
                if entry is not None and entry["element_type"] == ElementType.MULTI_WORD_STOP_WORD:
                    for i_tt in range(i_st, j + 1):
                        self.query_element[i_tt] = ""
                    return True
        return False

    # -------------------------------------------------------------------------------------
    # TaxmannQueryAnalizer.cs:702-735 - CheckKeyWordWithOutSpace
    # -------------------------------------------------------------------------------------
    def _check_keyword_without_space(self, qry_word: str) -> tuple[bool, str, str]:
        found = False
        search_text = ""
        key_text = ""
        for j in range(len(qry_word)):
            if qry_word[j].isdigit():
                key_wrd = qry_word[:j]
                param_str = qry_word[j:]
                entry = classify_token(key_wrd)
                if entry is not None and entry["element_type"] == ElementType.KEY_WORD:
                    key_text = key_wrd
                    if param_str[0] == "0":
                        search_text = (
                            f"{_get_key_search_text(key_wrd)} {param_str} | "
                            f"{_get_key_search_text(key_wrd)} {_remove_leading_zero(param_str)}"
                        )
                    else:
                        search_text = f"{_get_key_search_text(key_wrd)} {param_str}"
                    found = True
        return found, search_text, key_text

    # -------------------------------------------------------------------------------------
    # TaxmannQueryAnalizer.cs:914-966 - IsNextWordHighCourt
    # -------------------------------------------------------------------------------------
    def _is_next_word_high_court(self) -> tuple[bool, str]:
        back_track_count = 0
        next_word = self.get_element()
        back_track_count += 1
        if len(next_word.strip()) == 0:
            return False, ""
        while _is_stop_word(next_word) and not self.end_of_list:
            next_word = self.get_element()
            back_track_count += 1

        if next_word.replace(" ", "").upper() == "HIGHCOURT":
            return True, _get_key_search_text("HIGHCOURT")
        if next_word.replace(" ", "").upper() in ("TRIBUNAL", "ITAT"):
            return True, _get_key_search_text("TRIBUNAL")
        if next_word.upper() == "HIGH":
            next_word = self.get_element()
            back_track_count += 1
            while _is_stop_word(next_word) and not self.end_of_list:
                next_word = self.get_element()
                back_track_count += 1
            if next_word.upper() == "COURT":
                return True, " HIGH COURT"
            for _ in range(back_track_count):
                if len(next_word) > 0:
                    self.back_track()
            return False, ""
        for _ in range(back_track_count):
            if len(next_word) > 0:
                self.back_track()
        return False, ""

    # -------------------------------------------------------------------------------------
    # TaxmannQueryAnalizer.cs:1057-1190 - ProcessKeyWord
    # -------------------------------------------------------------------------------------
    def _process_key_word(self, key_word: str) -> tuple[bool, str]:
        next_word = self.get_element()
        if len(next_word.strip()) == 0:
            return False, ""

        if classify_token(next_word) is not None:
            self.back_track()
            return False, ""

        first_char_is_digit = next_word[:1].isdigit()
        roman_value = _convert_roman_number(next_word) if not first_char_is_digit else 0
        if not first_char_is_digit and roman_value > 0:
            key_search = _get_key_search_text(key_word)
            if "|" in key_search:
                parts = key_search.split("|")
                st = "|".join(f"{p.strip()} {next_word}" for p in parts)
                search_text = st
            else:
                search_text = f"{key_search} {next_word}"
            return True, search_text
        if not first_char_is_digit and roman_value <= 0:
            # TaxmannQueryAnalizer.cs:1099-1102: falls through to BackTrack with no result.
            if not next_word[:1].isdigit():
                self.back_track()
                return False, ""

        # next_word starts with a digit (line 1108 onward).
        next_word2 = ""
        for space_variant, plain_variant in (
            (" I", "I"), (" i", "i"), (" O", "O"), (" o", "o"),
            (" P", "P"), (" p", "p"),
        ):
            if space_variant in next_word:
                next_word2 = next_word.replace(space_variant, plain_variant)
                break
            if plain_variant in next_word:
                next_word2 = next_word.replace(plain_variant, " " + plain_variant)
                break

        key_word_search = _get_key_search_text(key_word)
        if next_word[0] == "0":
            search_text = (
                f"{key_word_search} {next_word} | "
                f"{key_word_search} {_remove_leading_zero(next_word)}"
            )
            if next_word2:
                search_text += (
                    f" | {key_word_search} {next_word2} | "
                    f"{key_word_search} {_remove_leading_zero(next_word2)}"
                )
            return True, search_text

        if "ICDS |" in key_word_search and next_word.isdigit():
            new_next_word = _numeric_to_roman(int(next_word))
            entry = classify_token(key_word)
            if entry is not None and "|" in (entry["search_text"] or ""):
                parts = key_word_search.split("|")
                st = "|".join(f"{p.strip()} {new_next_word}" for p in parts)
                return True, st
            return False, ""

        if "|" in key_word_search:
            parts = key_word_search.split("|")
            st = "|".join(
                f"{p.strip()} {next_word} | {p.strip()} 0{next_word}" for p in parts
            )
        else:
            st = f"{key_word_search} {next_word} | {key_word_search} 0{next_word}"
        if next_word2:
            st += f" | {key_word_search} {next_word2} | {key_word_search} 0{next_word2}"
        return True, st

    # -------------------------------------------------------------------------------------
    # TaxmannQueryAnalizer.cs:1191-1238 - ProcessKeyWordType2
    # -------------------------------------------------------------------------------------
    def _process_key_word_type2(self, key_word: str) -> tuple[bool, str, str]:
        next_word = self.get_element()
        if len(next_word.strip()) > 0:
            if classify_token(next_word) is not None:
                self.back_track()
                return False, "", ""
            if next_word[:1].isdigit():
                st_element = _get_key_search_text(key_word).split(";")
                return True, f"{st_element[0]} {next_word}", st_element[1]
            st_element = _get_key_search_text(key_word).split(";")
            self.back_track()
            return False, st_element[0], st_element[1]
        st_element = _get_key_search_text(key_word).split(";")
        return False, st_element[0], st_element[1]

    # -------------------------------------------------------------------------------------
    # TaxmannQueryAnalizer.cs:1239-1289 - ProcessKeyWordHighCourt
    # -------------------------------------------------------------------------------------
    def _process_key_word_high_court(self, key_word: str) -> tuple[bool, str]:
        back_track_count = 0
        next_word = self.get_element()
        back_track_count += 1
        if len(next_word.strip()) == 0:
            return False, ""
        while _is_stop_word(next_word) and not self.end_of_list:
            next_word = self.get_element()
            back_track_count += 1

        # Line 1252: `GetResource(NextWord.Replace(" ", "")).ToUpper()` - .ToUpper() applies to
        # the RETURN VALUE, not to next_word before lookup. This means the lookup here is
        # case-sensitive on next_word, unlike IsNextWordHighCourt's correctly-normalized check
        # at line 929. Preserved verbatim - verified by reading, not a guess.
        entry = classify_token_case_sensitive(next_word.replace(" ", ""))
        if entry is not None and entry["element_type"] == ElementType.COURT:
            key_items = _get_key_search_text(key_word).split("|")
            search_text = ""
            for item in key_items:
                if search_text.strip():
                    search_text += f" | {item} {next_word}"
                else:
                    search_text = f"{item} {next_word}"
            return True, search_text
        for _ in range(back_track_count):
            if len(next_word) > 0:
                self.back_track()
        return False, ""

    # -------------------------------------------------------------------------------------
    # TaxmannQueryAnalizer.cs:1290-1319 - ProcessKeyWordDated
    # -------------------------------------------------------------------------------------
    def _process_key_word_dated(self) -> tuple[bool, str, date | None]:
        next_word = self.get_element()
        if len(next_word.strip()) == 0:
            return False, "", None
        if is_date_format(next_word.upper()):
            return True, TokenType.DATE_FMT, get_date_format(next_word)
        ok, result_date = is_month_format(next_word, self)
        if ok:
            return True, TokenType.MONTH_FMT, result_date
        self.back_track()
        return False, "", None

    # -------------------------------------------------------------------------------------
    # TaxmannQueryAnalizer.cs:1320-1501 - ProcessCitation
    # -------------------------------------------------------------------------------------
    def _process_citation(self, entry_point: str, partial_citation: str) -> tuple[bool, str]:
        next_word = self.get_element()
        if entry_point == "Y":
            if next_word.isdigit():
                ok, citation = self._process_citation("V", f"{partial_citation} {next_word}")
                if ok:
                    return True, citation
                if len(next_word) > 0:
                    self.back_track()
                return False, ""
            if len(next_word) > 0:
                self.back_track()
            return False, ""

        if entry_point == "V":
            entry = classify_token(next_word)
            if entry is not None and entry["element_type"] == ElementType.JOURNAL:
                ok, citation = self._process_citation("J", f"{partial_citation} {next_word}")
                if ok:
                    return True, citation
                return True, f"{partial_citation} {next_word}"
            if len(next_word) > 0:
                self.back_track()
            return False, ""

        if entry_point == "J":
            if next_word.isdigit():
                ok, citation = self._process_citation("P", f"{partial_citation} {next_word}")
                if ok:
                    return True, citation
                return True, f"{partial_citation} {next_word}"
            entry = classify_token(next_word)
            if entry is not None:
                if entry["element_type"] == ElementType.COURT:
                    ok, citation = self._process_citation("C", f"{partial_citation} {next_word}")
                    if ok:
                        return True, citation
                    return True, f"{partial_citation} {next_word}"
                if len(next_word) > 0:
                    self.back_track()
                return True, partial_citation
            if len(next_word) > 0:
                self.back_track()
            return True, partial_citation

        if entry_point == "P":
            entry = classify_token(next_word)
            if entry is not None:
                if entry["element_type"] == ElementType.COURT:
                    ok, citation = self._process_citation("C", f"{partial_citation} {next_word}")
                    if ok:
                        return True, citation
                    return True, f"{partial_citation} {next_word}"
                if entry["element_type"] == ElementType.SBTM:
                    return True, f"{partial_citation} {next_word}"
                if len(next_word) > 0:
                    self.back_track()
                return True, partial_citation
            if len(next_word) > 0:
                self.back_track()
            return True, partial_citation

        if entry_point == "C":
            entry = classify_token(next_word)
            if entry is not None:
                if entry["element_type"] == ElementType.SBTM:
                    return True, f"{partial_citation} {next_word}"
                if len(next_word) > 0:
                    self.back_track()
                return True, partial_citation
            if next_word.isdigit():
                ok, citation = self._process_citation("P", f"{partial_citation} {next_word}")
                if ok:
                    return True, citation
                return True, f"{partial_citation} {next_word}"
            if len(next_word) > 0:
                self.back_track()
            return True, partial_citation

        if len(next_word) > 0:
            self.back_track()
        return False, ""

    # -------------------------------------------------------------------------------------
    # TaxmannQueryAnalizer.cs:967-1056 - ProcessDate
    # -------------------------------------------------------------------------------------
    def _process_date(self, partial_string: str) -> tuple[bool, date | None]:
        return process_date(partial_string, self)

    # -------------------------------------------------------------------------------------
    # TaxmannQueryAnalizer.cs:776-846 / 847-913 - IsDayMonthFormat / IsMonthFormat wrappers
    # -------------------------------------------------------------------------------------
    def _is_day_month_format(self, str_text: str) -> tuple[bool, date | None]:
        return is_day_month_format(str_text, self)

    def _is_month_format(self, str_text: str) -> tuple[bool, date | None]:
        return is_month_format(str_text, self)

    # -------------------------------------------------------------------------------------
    # TaxmannQueryAnalizer.cs:379-456 - NewToken / CreateToken / PushToken
    # -------------------------------------------------------------------------------------
    @staticmethod
    def _new_token() -> RepotaxmannapiToken:
        return RepotaxmannapiToken(
            query_text="", query_date=None, org_text="", type="",
            or_in=False, proximity=ProximityDefault.DEFAULT_VALUE, group_id="0",
        )

    @staticmethod
    def _create_token(
        buffer_obj: object, t_type: str, proximity: int, set_or_in: bool = False,
        group_id: str = "0",
    ) -> RepotaxmannapiToken:
        query_date: date | None = None
        if t_type == TokenType.DATE_FMT:
            assert isinstance(buffer_obj, date)
            query_date = buffer_obj
            query_text = query_date.strftime("%d %m %Y")
        elif t_type == TokenType.MONTH_FMT:
            assert isinstance(buffer_obj, date)
            query_date = buffer_obj
            query_text = f"{month_names()[query_date.month]} {query_date.year}"
        elif t_type == TokenType.DATE_MONTH_FMT:
            assert isinstance(buffer_obj, date)
            query_date = buffer_obj
            query_text = f"{query_date.day:02d} {month_names()[query_date.month]}"
        else:
            query_text = str(buffer_obj)
        return RepotaxmannapiToken(
            query_text=query_text, query_date=query_date, org_text="", type=t_type,
            or_in=set_or_in, proximity=proximity, group_id=group_id,
        )

    def _push_token(self, tokens: list[RepotaxmannapiToken], token: RepotaxmannapiToken) -> None:
        # TaxmannQueryAnalizer.cs:441-456
        if token.type == TokenType.DATE_FMT or len(token.query_text) > 0:
            tokens.append(token)

    # -------------------------------------------------------------------------------------
    # TaxmannQueryAnalizer.cs:1542-1999 - ProcessorQuery (the main tokenization loop)
    # -------------------------------------------------------------------------------------
    def process_query(self) -> list[RepotaxmannapiToken]:
        # No early `if self.end_of_list: return tokens` here - a fully-quoted query (e.g.
        # `"section 52"`) strips every word into phrase_elements during __init__, leaving
        # query_element empty and end_of_list True from the start. The main while loop below
        # already no-ops correctly in that case (its own `while not self.end_of_list`
        # condition is simply never true); an early return here would skip past it AND past
        # the phrase_elements-append tail below (TaxmannQueryAnalizer.cs:1985-1993, which the
        # real source runs unconditionally after the main loop) - silently discarding every
        # token for a wholly-quoted query and leaving `should` with nothing but the always-on
        # static group boosts. Confirmed real bug, not by inspection alone: reproduced via
        # /v1/query-analysis trace for `"section 52"` and `"financial management"` - both
        # produced the exact same query.bool.should (6 static group-boost `term` clauses,
        # zero `match_phrase` clauses referencing the query text at all), and ranked purely by
        # group/documenttypeboost/viewcount/recency, explaining why irrelevant sections (e.g.
        # "Short title, extent and commencement") outranked the actual query text.
        tokens: list[RepotaxmannapiToken] = []
        temp_token = self._new_token()

        while not self.end_of_list:
            found, r_text, i_mat_pos = self._check_for_keyword(self.i_ptr)
            if found:
                self.i_ptr = i_mat_pos + 1
                entry = classify_token(r_text)
                assert entry is not None
                element_code = entry["element_type"]

                if element_code == ElementType.KEY_WORD_ONLY:
                    # TaxmannQueryAnalizer.cs:1594-1601. No dictionary entries of this type
                    # exist in the extracted table (untestable/unreachable with current data).
                    self._push_token(tokens, temp_token)
                    temp_token = self._new_token()
                    self._push_token(
                        tokens,
                        self._create_token(
                            _get_key_search_text(r_text), TokenType.KEY_WORD,
                            _get_key_proximity(r_text), group_id=entry["group_id"],
                        ),
                    )
                    if r_text.upper().startswith("ARTICLE"):  # cs:1600
                        self.is_article = True

                elif element_code == ElementType.KEY_WORD:
                    ok, s_result = self._process_key_word(r_text)
                    if ok:
                        self._push_token(tokens, temp_token)
                        temp_token = self._new_token()
                        self._push_token(
                            tokens,
                            self._create_token(
                                s_result, TokenType.SECTION_TYPE_FORMAT,
                                _get_key_proximity(r_text), group_id=entry["group_id"],
                            ),
                        )
                    elif r_text.upper() != "AS":
                        self._push_token(tokens, temp_token)
                        temp_token = self._new_token()
                        self._push_token(
                            tokens,
                            self._create_token(
                                _get_key_search_text(r_text), TokenType.KEY_WORD,
                                _get_key_proximity(r_text), group_id=entry["group_id"],
                            ),
                        )
                        if r_text.upper().startswith("ARTICLE"):  # cs:1621
                            self.is_article = True

                elif element_code == ElementType.KEY_WORD_TYPE2:
                    ok, s_result, oth_text = self._process_key_word_type2(r_text)
                    self._push_token(tokens, temp_token)
                    temp_token = self._new_token()
                    if ok:
                        self._push_token(
                            tokens,
                            self._create_token(
                                s_result, TokenType.SECTION_TYPE_FORMAT,
                                _get_key_proximity(r_text), group_id=entry["group_id"],
                            ),
                        )
                        self._push_token(
                            tokens,
                            self._create_token(
                                oth_text, TokenType.KEY_WORD, 1, group_id=entry["group_id"],
                            ),
                        )
                    else:
                        self._push_token(
                            tokens,
                            self._create_token(
                                _get_key_search_text(r_text), TokenType.KEY_WORD,
                                _get_key_proximity(r_text), group_id=entry["group_id"],
                            ),
                        )
                        self._push_token(
                            tokens,
                            self._create_token(
                                oth_text, TokenType.KEY_WORD, 1, group_id=entry["group_id"],
                            ),
                        )
                        if r_text.upper().startswith("ARTICLE"):  # cs:1645
                            self.is_article = True

                elif element_code == ElementType.KEY_WORD_HIGH_COURT:
                    ok, s_result = self._process_key_word_high_court(r_text)
                    self._push_token(tokens, temp_token)
                    temp_token = self._new_token()
                    if ok:
                        self._push_token(
                            tokens,
                            self._create_token(
                                s_result, TokenType.SECTION_TYPE_FORMAT,
                                _get_key_proximity(r_text), group_id=entry["group_id"],
                            ),
                        )
                    else:
                        self._push_token(
                            tokens,
                            self._create_token(
                                _get_key_search_text(r_text), TokenType.KEY_WORD,
                                _get_key_proximity(r_text), group_id=entry["group_id"],
                            ),
                        )

                elif element_code == ElementType.KEY_WORD_OR_STOP_WORD:
                    ok, s_result = self._process_key_word(r_text)
                    if ok:
                        self._push_token(tokens, temp_token)
                        temp_token = self._new_token()
                        self._push_token(
                            tokens,
                            self._create_token(
                                s_result, TokenType.SECTION_TYPE_FORMAT,
                                _get_key_proximity(r_text), group_id=entry["group_id"],
                            ),
                        )
                    # else: dropped entirely - no `else` branch in the C# (1667-1677).

                elif element_code == ElementType.ZONE:
                    self._push_token(tokens, temp_token)
                    temp_token = self._new_token()
                    self._push_token(
                        tokens,
                        self._create_token(
                            _get_key_search_text(r_text), TokenType.SEARCH_ZONE,
                            _get_key_proximity(r_text), group_id=entry["group_id"],
                        ),
                    )

                elif element_code == ElementType.NUM_ALPHA_ZONE:
                    self._push_token(tokens, temp_token)
                    temp_token = self._new_token()
                    self._push_token(
                        tokens,
                        self._create_token(
                            _get_key_search_text(r_text), TokenType.NUM_ALPHA_ZONE,
                            _get_key_proximity(r_text), group_id=entry["group_id"],
                        ),
                    )

                elif element_code == ElementType.COUNTRY:
                    self._push_token(tokens, temp_token)
                    temp_token = self._new_token()
                    self._push_token(
                        tokens,
                        self._create_token(
                            _get_key_search_text(r_text), TokenType.KEY_WORD,
                            _get_key_proximity(r_text), group_id=entry["group_id"],
                        ),
                    )
                    self.is_country = True  # cs:~1697

                elif element_code == ElementType.STOP_WORD:
                    pass

                elif element_code == ElementType.SYNONYM:
                    # TaxmannQueryAnalizer.cs:1701-1706. Unreachable with the extracted
                    # dictionary (zero "44" entries) - see SYNONYM_WORD_LIST note above.
                    self._push_token(tokens, temp_token)
                    temp_token = self._new_token()
                    self._push_token(
                        tokens,
                        self._create_token(
                            SYNONYM_WORD_LIST.get(r_text.upper(), ""), TokenType.SYNONYM,
                            _get_key_proximity(r_text), True, group_id=entry["group_id"],
                        ),
                    )

                elif element_code == ElementType.KEY_WORD_DATED:
                    ok, tk_type, result_date = self._process_key_word_dated()
                    if ok:
                        self._push_token(tokens, temp_token)
                        self._push_token(
                            tokens,
                            self._create_token(
                                result_date, tk_type, ProximityDefault.DATE_TYPE,
                                group_id=entry["group_id"],
                            ),
                        )
                        temp_token = self._new_token()

                else:
                    handled = False
                    if element_code == ElementType.COURT:
                        ok, s_result = self._is_next_word_high_court()
                        if ok:
                            self._push_token(tokens, temp_token)
                            temp_token = self._new_token()
                            if "|" in s_result:
                                key_items = s_result.split("|")
                                new_result = ""
                                for item in key_items:
                                    if new_result.strip():
                                        new_result += f" | {r_text} {item}"
                                    else:
                                        new_result = f"{r_text} {item}"
                                self._push_token(
                                    tokens,
                                    self._create_token(
                                        new_result, TokenType.SEARCH_ZONE, 3,
                                        group_id=entry["group_id"],
                                    ),
                                )
                            else:
                                self._push_token(
                                    tokens,
                                    self._create_token(
                                        f"{r_text} {s_result}", TokenType.SEARCH_ZONE, 3,
                                        group_id=entry["group_id"],
                                    ),
                                )
                            handled = True
                    elif element_code == ElementType.JOURNAL:
                        ok, s_result = self._process_citation("J", r_text)
                        if ok:
                            self._push_token(tokens, temp_token)
                            self._push_token(
                                tokens,
                                self._create_token(
                                    s_result, TokenType.CITATION,
                                    _get_key_proximity("CITATION"),
                                    group_id=classify_token("CITATION")["group_id"],
                                ),
                            )
                            temp_token = self._new_token()
                            handled = True

                    if not handled:
                        if temp_token.type in (TokenType.NUMBER, TokenType.DATE_FMT):
                            self._push_token(tokens, temp_token)
                            temp_token = self._new_token()
                            temp_token.type = TokenType.TEXT
                            temp_token.proximity = ProximityDefault.TEXT
                        elif temp_token.type == TokenType.TEXT:
                            pass
                        else:
                            temp_token = self._new_token()
                            temp_token.type = TokenType.TEXT
                            temp_token.proximity = ProximityDefault.TEXT
                        key_search = _get_key_search_text(r_text)
                        if "|" in key_search:
                            temp_token.query_text = f"{temp_token.query_text} {r_text}".strip()
                        else:
                            temp_token.query_text = f"{temp_token.query_text} {key_search}".strip()

            else:
                qry_wrd = self.query_element[self.i_ptr]
                self.i_ptr += 1
                is_no = qry_wrd.lstrip("+-").isdigit() and qry_wrd.strip() != ""
                if is_no and " " in qry_wrd:
                    is_no = False

                if is_no:
                    ok, s_result = self._process_citation("Y", qry_wrd)
                    if ok:
                        self._push_token(tokens, temp_token)
                        self._push_token(
                            tokens,
                            self._create_token(
                                s_result, TokenType.CITATION, _get_key_proximity("CITATION"),
                                group_id=classify_token("CITATION")["group_id"],
                            ),
                        )
                        temp_token = self._new_token()
                        continue
                    ok, s_result = self._process_citation("V", qry_wrd)
                    if ok:
                        self._push_token(tokens, temp_token)
                        self._push_token(
                            tokens,
                            self._create_token(
                                s_result, TokenType.CITATION, _get_key_proximity("CITATION"),
                                group_id=classify_token("CITATION")["group_id"],
                            ),
                        )
                        temp_token = self._new_token()
                        continue
                    ok, result_date = self._process_date(qry_wrd)
                    if ok:
                        self._push_token(tokens, temp_token)
                        self._push_token(
                            tokens,
                            self._create_token(
                                result_date, TokenType.DATE_FMT, ProximityDefault.DATE_TYPE,
                            ),
                        )
                        temp_token = self._new_token()
                        continue
                    ok, result_date = self._is_day_month_format(qry_wrd)
                    if ok:
                        self._push_token(tokens, temp_token)
                        self._push_token(
                            tokens,
                            self._create_token(
                                result_date, TokenType.DATE_MONTH_FMT,
                                ProximityDefault.DATE_TYPE,
                            ),
                        )
                        temp_token = self._new_token()
                        continue

                    if temp_token.type == TokenType.NUMBER:
                        pass
                    elif temp_token.type in (TokenType.DATE_FMT, TokenType.TEXT):
                        self._push_token(tokens, temp_token)
                        temp_token = self._new_token()
                        temp_token.type = TokenType.NUMBER
                        temp_token.proximity = ProximityDefault.NUMERIC
                    else:
                        temp_token = self._new_token()
                        temp_token.type = TokenType.NUMBER
                        temp_token.proximity = ProximityDefault.NUMERIC
                    if temp_token.query_text != qry_wrd:
                        temp_token.query_text = f"{temp_token.query_text} {qry_wrd}".strip()
                    elif len(self.query_element) > self.i_ptr:
                        self.i_ptr += 1

                elif _is_stop_word(qry_wrd):
                    pass

                elif is_date_format(qry_wrd):
                    self._push_token(tokens, temp_token)
                    self._push_token(
                        tokens,
                        self._create_token(
                            get_date_format(qry_wrd), TokenType.DATE_FMT,
                            ProximityDefault.DATE_TYPE,
                        ),
                    )
                    temp_token = self._new_token()

                elif (month_ok_result := is_month_format(qry_wrd, self))[0]:
                    self._push_token(tokens, temp_token)
                    self._push_token(
                        tokens,
                        self._create_token(
                            month_ok_result[1], TokenType.MONTH_FMT, ProximityDefault.DATE_TYPE,
                        ),
                    )
                    temp_token = self._new_token()

                elif (kw_result := self._check_keyword_without_space(qry_wrd))[0]:
                    _, s_result, s_match_key = kw_result
                    self._push_token(tokens, temp_token)
                    temp_token = self._new_token()
                    match_entry = classify_token(s_match_key)
                    self._push_token(
                        tokens,
                        self._create_token(
                            s_result, TokenType.SECTION_TYPE_FORMAT,
                            _get_key_proximity(s_match_key),
                            group_id=(match_entry or {}).get("group_id", "0"),
                        ),
                    )

                elif _is_cir_no_format(qry_wrd):
                    self._push_token(tokens, temp_token)
                    self._push_token(
                        tokens,
                        self._create_token(
                            qry_wrd, TokenType.NOTIFICATION_NUMBER,
                            ProximityDefault.NOTIFICATION,
                        ),
                    )
                    temp_token = self._new_token()
                    temp_token.type = TokenType.TEXT
                    temp_token.proximity = ProximityDefault.TEXT

                else:
                    ok, s_result = _is_number_in_words_type(qry_wrd)
                    if ok and len(s_result) > 0:
                        qry_wrd = s_result

                    if temp_token.type in (TokenType.NUMBER, TokenType.DATE_FMT):
                        self._push_token(tokens, temp_token)
                        temp_token = self._new_token()
                        temp_token.type = TokenType.TEXT
                        temp_token.proximity = ProximityDefault.TEXT
                    elif temp_token.type == TokenType.TEXT:
                        pass
                    else:
                        temp_token = self._new_token()
                        temp_token.type = TokenType.TEXT
                        temp_token.proximity = ProximityDefault.TEXT
                    temp_token.query_text = f"{temp_token.query_text} {qry_wrd}".strip()

        if len(temp_token.type) > 0:
            self._push_token(tokens, temp_token)
            temp_token = self._new_token()

        # TaxmannQueryAnalizer.cs:1985-1993 - phrase words are appended LAST, after every
        # token from the main scanning loop, regardless of where they appeared in the query.
        for phrase in self.phrase_elements:
            self._push_token(
                tokens,
                self._create_token(phrase, TokenType.PHRASE_WORD, ProximityDefault.PHRASE),
            )

        if self.is_global and self.is_article and not self.is_country:
            # TaxmannQueryAnalizer.cs:1968-1984: rewrite QueryText/Proximity (NOT Type) to
            # the "EXPERTSOPINION" dictionary entry's own values for every token whose
            # QueryText starts with "ARTICLE", and reset that token's group_id to
            # EXPERTSOPINION's own group_id (this port's per-token equivalent of the real
            # source's query-level ReSetPrimaryTag("EXPERTSOPINION") - see this module's own
            # docstring on the per-token group_id deviation).
            expertsopinion_text = _get_key_search_text("EXPERTSOPINION")
            expertsopinion_proximity = _get_key_proximity("EXPERTSOPINION")
            expertsopinion_entry = classify_token("EXPERTSOPINION")
            expertsopinion_group_id = (
                expertsopinion_entry["group_id"] if expertsopinion_entry else "0"
            )
            for i, tok in enumerate(tokens):
                if tok.query_text.upper().startswith("ARTICLE"):
                    tokens[i] = RepotaxmannapiToken(
                        query_text=expertsopinion_text, query_date=tok.query_date,
                        org_text=tok.org_text, type=tok.type, or_in=tok.or_in,
                        proximity=expertsopinion_proximity, group_id=expertsopinion_group_id,
                    )

        return tokens


def classify_token_case_sensitive(word: str) -> TokenDictEntry | None:
    """Used only by ProcessKeyWordHighCourt's exact port (TaxmannQueryAnalizer.cs:1252) to
    preserve its real case-sensitivity bug - see _process_key_word_high_court's comment. The
    extracted dictionary is keyed by uppercase form only (Task 2), so a non-uppercase `word`
    here always misses, exactly like the original's un-normalized GetResource(word) call
    would against uppercase-only resource keys."""
    return load_repotaxmannapi_token_dictionary().get(word) if word.isupper() else None


def tokenize(query: str, is_global: bool = False) -> list[RepotaxmannapiToken]:
    """Entry point: full port of `new TaxmannQueryAnalizer(query)` followed by a single
    `ProcessorQuery()` call (TaxmannQueryAnalizer.cs:114-193, 1542-1999). `is_global`
    (2026-09-06): threads through the real source's `isGlobalSearch` field - gates the
    ARTICLE->EXPERTSOPINION remap pass (TaxmannQueryAnalizer.cs:1968-1984). Defaults to
    False so every existing caller keeps today's behavior unchanged; only `es_client.py`'s
    repotaxmannapi global-search path passes True."""
    analyzer = _Analyzer(query, is_global=is_global)
    return analyzer.process_query()


def is_whole_query_exact_phrase(query: str) -> bool:
    """True when the ENTIRE search-bar query is one double-quoted phrase (e.g.
    `"section 52"`) - no unquoted words alongside it. Shared predicate for the exact-phrase
    fast path: `es_client._build_repotaxmannapi_field_query` uses this same condition to skip
    FunctionScore, and Instant mode's `run_instant` uses it to skip the ML shape classifier
    and Milvus entirely - a fully-quoted query is an unambiguous exact-lookup request, not
    something a KEYWORD/HYBRID/INTENT shape classification or semantic (dense-vector) search
    adds any value to. False for an empty/all-whitespace query (nothing to search) and for a
    mixed query (unquoted words alongside a quoted phrase) - no real-source evidence exists
    yet for how repotaxmannapi routes that mixed shape, so it stays on the normal path."""
    tokens = tokenize(query)
    return bool(tokens) and all(t.type == TokenType.PHRASE_WORD for t in tokens)
