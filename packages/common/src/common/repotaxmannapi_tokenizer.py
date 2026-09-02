"""Python port of repotaxmannapi/TaxmannAPI/Elastic/TaxmannQueryAnalizer.cs (a separate,
read-only .NET codebase) - see docs/superpowers/specs/2026-09-01-repotaxmannapi-exact-replica-design.md
for why this exists as a parallel, opt-in path rather than replacing this repo's existing
(eval-verified, sum-mode) tokenizer."""
from common.repotaxmannapi_token_dictionary import TokenDictEntry, load_repotaxmannapi_token_dictionary


def classify_token(word: str) -> TokenDictEntry | None:
    """Direct port of TaxmannQueryAnalizer.cs's GetResource(word.ToUpper()) lookup
    (TaxmannQueryAnalizer.cs:217-233) - the C# returns "" for a missing key, which
    SetPrimaryTag (lines 296-314) treats as "no dictionary entry"; this returns None for
    the same case so callers use a Pythonic None-check instead of an empty-string check."""
    return load_repotaxmannapi_token_dictionary().get(word.upper())
