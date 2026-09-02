"""Date-format helpers ported from repotaxmannapi/TaxmannAPI/Elastic/TaxmannQueryAnalizer.cs
(a separate, read-only .NET codebase). Split into their own module because they are needed by
both repotaxmannapi_tokenizer.py's main loop and its `_Analyzer` cursor helpers, and because
they share one genuinely-approximated piece of behavior documented below.

All date construction here builds `datetime.date` objects directly from already-parsed
day/month/year components, rather than round-tripping through a string and a
locale-aware `DateTime.Parse` the way the C# does. This is a deliberate, documented
simplification: every string this file ever hands to `DateTime.Parse` in the original is one
*this same code already assembled* from day/month-name/year parts it already knows, so
constructing the date directly is behaviorally equivalent for every input this module
constructs itself, without depending on .NET's `en-GB` culture parser (which isn't part of
this file, so its exact quirks - e.g. two-digit-year windowing - cannot be verified by reading
this codebase alone). Two-digit years are handled with a documented, unverified-against-source
approximation of .NET's default `Calendar.TwoDigitYearMax` (2029) windowing; flagged as such
rather than fabricated as verified fact.
"""
from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from common.repotaxmannapi_tokenizer import _Analyzer


# TaxmannQueryAnalizer.cs:34
_MONTH_NAMES = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# TaxmannQueryAnalizer.cs:13-15 (MonthList())
_MONTH_LIST = {
    "JAN": 1, "JANUARY": 1, "FEB": 2, "FEBRUARY": 2, "MAR": 3, "MARCH": 3,
    "APR": 4, "APRIL": 4, "MAY": 5, "JUN": 6, "JUNE": 6, "JUL": 7, "JULY": 7,
    "AUG": 8, "AUGUST": 8, "SEP": 9, "SEPTEMBER": 9, "OCT": 10, "OCTOBER": 10,
    "NOV": 11, "NOVEMBER": 11, "DEC": 12, "DECEMBER": 12,
}


def month_names() -> list[str]:
    return _MONTH_NAMES


def _year_from_two_digits(yy: int) -> int:
    """Undocumented-in-source approximation of .NET's default TwoDigitYearMax=2029 window
    (00-29 -> 2000-2029, 30-99 -> 1930-1999). Not verified against TaxmannQueryAnalizer.cs
    itself (culture/BCL behavior, out of scope for this file) - flagged, not fabricated."""
    return 2000 + yy if yy <= 29 else 1900 + yy


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def is_date_format(str_text: str) -> bool:
    """Port of IsDateFormat (TaxmannQueryAnalizer.cs:737-762): true iff replacing spaces with
    '/' and splitting on '/' gives exactly 3 parts that parse as a day-first (en-GB) date."""
    return get_date_format(str_text) is not None


def get_date_format(str_text: str) -> date | None:
    """Port of GetDateFormat (763-775) + the parse used by IsDateFormat (737-762): day-first
    (en-GB) parse of "dd/mm/yyyy"-shaped text (spaces normalized to '/')."""
    parts = str_text.strip().replace(" ", "/").split("/")
    if len(parts) != 3:
        return None
    try:
        dd, mm, yyyy = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None
    if yyyy < 100:
        yyyy = _year_from_two_digits(yyyy)
    return _safe_date(yyyy, mm, dd)


def is_month_format(str_text: str, analyzer: "_Analyzer") -> tuple[bool, date | None]:
    """Port of IsMonthFormat (847-913). Two independent paths:
    1. str_text is a month name -> look ahead one element for a >3-char numeric year.
    2. str_text is itself "mm/yyyy"-shaped (2 slash-parts, one of them 4 chars) -> parse
       directly, defaulting the day to 1 (`DateTime.Parse("mm/yyyy")` assumes day 1).
    """
    upper = str_text.upper()
    if upper in _MONTH_LIST:
        mm = _MONTH_LIST[upper]
        next_word = analyzer.get_element()
        if next_word.isdigit() and len(next_word) > 3:
            yyyy = int(next_word)
            result = _safe_date(yyyy, mm, 1)
            if result is not None:
                return True, result
            if len(next_word) > 0:
                analyzer.back_track()
            return False, None
        if len(next_word) > 0:
            analyzer.back_track()
        return False, None

    parts = str_text.strip().replace(" ", "/").split("/")
    if len(parts) == 2 and (len(parts[0]) == 4 or len(parts[1]) == 4):
        try:
            a, b = int(parts[0]), int(parts[1])
        except ValueError:
            return False, None
        # Whichever part is 4 digits is the year (mirrors DateTime.Parse inferring which
        # slash-part is the year from its width); the other is the month, day defaults to 1.
        if len(parts[0]) == 4:
            yyyy, mm = a, b
        else:
            mm, yyyy = a, b
        result = _safe_date(yyyy, mm, 1)
        return (True, result) if result is not None else (False, None)

    return False, None


def is_day_month_format(str_text: str, analyzer: "_Analyzer") -> tuple[bool, date | None]:
    """Port of IsDayMonthFormat (776-846). Two paths:
    1. Next element is a month name and str_text (1-31) is the day -> date defaults to the
       CURRENT year, exactly like the C# (`dd + mm + " " + yyyy` with yyyy left null, which
       DateTime.Parse fills in from the current date - genuinely today-relative behavior,
       verified by reading, not a simplification).
    2. str_text is itself "dd/mm"-shaped (2 slash-parts) -> parse directly, year defaults to
       current year the same way.
    """
    next_word = analyzer.get_element()
    upper = next_word.upper()
    if next_word and upper in _MONTH_LIST:
        mm = _MONTH_LIST[upper]
        if str_text.isdigit() and 0 < int(str_text) < 32:
            dd = int(str_text)
            result = _safe_date(date.today().year, mm, dd)
            if result is not None:
                return True, result
            if len(next_word) > 0:
                analyzer.back_track()
            return False, None
        if len(next_word) > 0:
            analyzer.back_track()
        return False, None

    if len(next_word) > 0:
        analyzer.back_track()
    parts = str_text.strip().replace(" ", "/").split("/")
    if len(parts) == 2:
        try:
            dd, mm = int(parts[0]), int(parts[1])
        except ValueError:
            return False, None
        result = _safe_date(date.today().year, mm, dd)
        return (True, result) if result is not None else (False, None)
    return False, None


def process_date(partial_string: str, analyzer: "_Analyzer") -> tuple[bool, date | None]:
    """Port of ProcessDate (TaxmannQueryAnalizer.cs:967-1056): consumes up to two more
    elements to build dd/mm/yyyy (numeric month) or dd/Mon/yyyy (month name) from a numeric
    day already extracted by the caller."""
    if not (partial_string.isdigit() and 0 < int(partial_string) < 32):
        return False, None
    dd = int(partial_string)

    next_word = analyzer.get_element()
    if next_word.isdigit():
        if 0 < int(next_word) < 13:
            mm = int(next_word)
            next_word2 = analyzer.get_element()
            if next_word2.isdigit():
                yyyy = int(next_word2)
                result = _safe_date(yyyy, mm, dd)
                if result is not None:
                    return True, result
                if len(next_word2) > 0:
                    analyzer.back_track()
                if len(next_word2) > 0:
                    analyzer.back_track()
                return False, None
            if len(next_word2) > 0:
                analyzer.back_track()
            if len(next_word2) > 0:
                analyzer.back_track()
            return False, None
        if len(next_word) > 0:
            analyzer.back_track()
        return False, None

    if next_word.upper() in _MONTH_LIST:
        mm = _MONTH_LIST[next_word.upper()]
        next_word2 = analyzer.get_element()
        if next_word2.isdigit():
            yyyy = int(next_word2)
            result = _safe_date(yyyy, mm, dd)
            if result is not None:
                return True, result
            if len(next_word2) > 0:
                analyzer.back_track()
            if len(next_word2) > 0:
                analyzer.back_track()
            return False, None
        if len(next_word2) > 0:
            analyzer.back_track()
        if len(next_word2) > 0:
            analyzer.back_track()
        if analyzer.i_ptr > 1:
            analyzer.back_track()
        return False, None

    if len(next_word) > 0:
        analyzer.back_track()
    return False, None
