DEFAULT_ELBOW_RATIO = 0.6
DEFAULT_MIN_KEEP = 1


def elbow_cutoff(
    scores_desc: list[float],
    min_keep: int = DEFAULT_MIN_KEEP,
    max_keep: int | None = None,
    ratio: float = DEFAULT_ELBOW_RATIO,
) -> int:
    """How many top-scored rows to keep: extend past min_keep while each next
    score stays within `ratio` of the previous one, so a single dominant
    match yields min_keep rows and a flat distribution yields up to max_keep
    (or the full list, if max_keep is None) rather than a fixed count either way."""
    count = min(min_keep, len(scores_desc))
    limit = len(scores_desc) if max_keep is None else min(len(scores_desc), max_keep)
    for i in range(count, limit):
        if scores_desc[i] < ratio * scores_desc[i - 1]:
            break
        count += 1
    return count
