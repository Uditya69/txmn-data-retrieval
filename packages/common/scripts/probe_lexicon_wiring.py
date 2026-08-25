"""Generate test queries for every wired legal_lexicon.json mapping (synonyms + normalizations)
and show what each one actually does through the real query-building pipeline.

Queries are derived straight from the lexicon data, not hand-typed - always in sync with
legal_lexicon.json, and one query per distinct target rather than per source key (many keys
share the same target, e.g. RULENO/RULES/RULESNO all -> RULE; testing one is enough).

Usage: see README at bottom of file / run command in the chat that generated this.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from common.query_tokenizer import (
    chunk_query, detect_group_signals, expand_query_normalizations, expand_query_synonyms,
)

_LEXICON_PATH = Path(__file__).parent.parent / "src" / "common" / "data" / "legal_lexicon.json"


def _load_lexicon() -> dict:
    return json.loads(_LEXICON_PATH.read_text(encoding="utf-8"))


def _probe_query(query: str) -> dict:
    synonym_expanded = expand_query_synonyms(query)
    full_expanded = expand_query_normalizations(synonym_expanded)
    chunks = chunk_query(query)
    signals = detect_group_signals(chunks)
    return {
        "query": query,
        "expanded": full_expanded if full_expanded != query else None,
        "group_signals": sorted(signals) if signals else None,
    }


def _print_row(label: str, sample_key: str, result: dict) -> None:
    changed = "CHANGED " if result["expanded"] or result["group_signals"] else "no-op   "
    print(f"{changed} {label:<18} key={sample_key:<22} query={result['query']!r}")
    if result["expanded"]:
        print(f"           -> expanded: {result['expanded']!r}")
    if result["group_signals"]:
        print(f"           -> group signal(s): {result['group_signals']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--section", choices=["normalizations", "synonyms", "both"], default="both",
        help="Which lexicon dict to generate queries for",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap how many synonym queries to print (there are ~1100; omit for all)",
    )
    args = parser.parse_args()
    lexicon = _load_lexicon()

    if args.section in ("normalizations", "both"):
        normalizations = lexicon["normalizations"]
        single_word_by_target: dict[str, list[str]] = defaultdict(list)
        multi_word_count = 0
        for key, target in normalizations.items():
            if " " in target:
                multi_word_count += 1
            else:
                single_word_by_target[target].append(key)

        print(f"=== normalizations: {len(single_word_by_target)} distinct single-word targets ===")
        print("(one query per target - other source keys for the same target aren't re-tested)\n")
        for target in sorted(single_word_by_target):
            sample_key = sorted(single_word_by_target[target])[0]
            query = f"regarding {sample_key.lower()} today"
            _print_row(target, sample_key, _probe_query(query))

        print(f"\n({multi_word_count} multi-word Act-name-expansion targets exist but are NOT wired - "
              f"e.g. normalizations['RESTRICTIVETRADEPRACTICESACT'] = "
              f"{normalizations.get('RESTRICTIVETRADEPRACTICESACT')!r}. Try one yourself: "
              f"probe_query('RESTRICTIVETRADEPRACTICESACT') should come back unchanged.)")

    if args.section in ("synonyms", "both"):
        synonyms = lexicon["synonyms"]
        keys = sorted(synonyms)
        if args.limit:
            keys = keys[: args.limit]
        print(f"\n=== synonyms: {len(synonyms)} entries, showing {len(keys)} ===\n")
        for key in keys:
            query = f"regarding {key.lower()} today"
            _print_row("(synonym)", key, _probe_query(query))


if __name__ == "__main__":
    main()
