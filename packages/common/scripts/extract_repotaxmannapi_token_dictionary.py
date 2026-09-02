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
