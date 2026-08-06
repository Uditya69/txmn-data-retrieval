import re

# doc_ids in this codebase are alphanumeric/hyphen/underscore tokens (e.g.
# "101010000000039445", "d1", "gold-doc"). A citation bracket may contain a
# single doc_id or a comma-separated list of them (e.g. "[d1, d2]"). This
# intentionally excludes multi-word prose asides like "[emphasis added]" or
# markdown links like "[Section 80C](...)" — those contain spaces or
# punctuation that isn't a valid doc_id token, so the whole bracket fails to
# match and is left alone rather than misread as a citation.
_CITATION_PATTERN = re.compile(r"\[([\w-]+(?:\s*,\s*[\w-]+)*)\]")


def _looks_like_doc_id(token: str) -> bool:
    # Real doc_ids in this codebase always contain at least one digit and are
    # at least 2 characters long (e.g. "101010000000039445", "d1", "999").
    # This excludes single alphabetic words like "sic" (from "[sic]",
    # common in legal-prose quoting) and lone-digit footnote markers like
    # "1" (from "[1]") without excluding realistic short doc_ids.
    return len(token) >= 2 and any(ch.isdigit() for ch in token)


def extract_cited_doc_ids(answer: str) -> set[str]:
    ids: set[str] = set()
    for match in _CITATION_PATTERN.findall(answer):
        for token in match.split(","):
            token = token.strip()
            if token and _looks_like_doc_id(token):
                ids.add(token)
    return ids


def validate_citations(answer: str, seen_doc_ids: set[str]) -> list[str]:
    return sorted(extract_cited_doc_ids(answer) - seen_doc_ids)
