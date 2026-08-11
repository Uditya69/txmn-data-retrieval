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
