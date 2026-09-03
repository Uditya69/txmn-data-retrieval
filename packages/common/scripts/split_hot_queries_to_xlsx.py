"""Splits Hot Queries/hot_search_queries.csv (already sorted count-desc) into
Hot Queries/hot_search_queries.xlsx with one tab per ~1M rows, since Excel's
per-sheet row cap (1,048,576) is smaller than the ~8M unique query CSV.

Run: uv run --package common python packages/common/scripts/split_hot_queries_to_xlsx.py
"""

import csv
import re
from pathlib import Path

from openpyxl import Workbook
from openpyxl.cell import WriteOnlyCell

_ROOT = Path(__file__).parent.parent.parent.parent
_CSV_PATH = _ROOT / "Hot Queries" / "hot_search_queries.csv"
_XLSX_PATH = _ROOT / "Hot Queries" / "hot_search_queries.xlsx"
_ROWS_PER_SHEET = 1_000_000

# openpyxl's own ILLEGAL_CHARACTERS_RE only covers control chars in the 0-31
# range and missed lone surrogates / other non-XML-1.0-valid chars present in
# this data, producing a malformed xlsx. Allowlist valid XML 1.0 codepoints
# instead of blacklisting known-bad ones. Built from ordinals via chr(), never
# typed as literal escape sequences, so nothing here can get mangled on save.
_XML_VALID_RANGES = [
    (9, 9),
    (10, 10),
    (13, 13),
    (32, 0xD7FF),
    (0xE000, 0xFFFD),
    (0x10000, 0x10FFFF),
]
_VALID_XML_CHARS_RE = re.compile(
    "[^" + "".join(chr(lo) + "-" + chr(hi) for lo, hi in _XML_VALID_RANGES) + "]"
)


def _sanitize(value: str) -> str:
    """Strip chars xlsx/XML 1.0 disallows - only for the Excel copy, the
    source CSV is untouched."""
    return _VALID_XML_CHARS_RE.sub("", value)


def _text_cell(ws, value: str) -> WriteOnlyCell:
    """Forces string data_type so Excel never guesses a leading '=', '+', '-'
    or '@' means a formula (that guess is what corrupted the first attempt -
    Excel wrote a <f> formula tag it then couldn't parse and deleted).
    WriteOnlyCell's parent must be the worksheet, not the workbook."""
    cell = WriteOnlyCell(ws, value=value)
    cell.data_type = "s"
    return cell


def main() -> None:
    wb = Workbook(write_only=True)
    sheet = None
    row_count = 0
    sheet_index = 0

    print(f"Reading {_CSV_PATH} ...")
    with _CSV_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if sheet is None or row_count >= _ROWS_PER_SHEET:
                sheet_index += 1
                sheet = wb.create_sheet(title=f"Sheet{sheet_index}")
                sheet.append(header)
                row_count = 0
                print(f"Writing Sheet{sheet_index} ...")
            sheet.append([_text_cell(sheet, _sanitize(row[0])), int(row[1])])
            row_count += 1

    print(f"Saving {_XLSX_PATH} ({sheet_index} sheets) ...")
    wb.save(_XLSX_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
