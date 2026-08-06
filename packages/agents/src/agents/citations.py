import re

_CITATION_PATTERN = re.compile(r"\[([^\[\]]+)\]")


def extract_cited_doc_ids(answer: str) -> set[str]:
    return set(_CITATION_PATTERN.findall(answer))


def validate_citations(answer: str, seen_doc_ids: set[str]) -> list[str]:
    return sorted(extract_cited_doc_ids(answer) - seen_doc_ids)
