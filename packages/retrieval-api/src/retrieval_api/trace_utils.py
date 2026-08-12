def ranked_trace(rows: list[dict], top_n: int = 10) -> list[dict]:
    """Point-1-to-last-point diagnostic for a single ranked list (e.g. ES): rank
    position + doc_id + score for the top N, so a gold doc_id's rank - or its
    absence - is visible straight in the trace without rerunning the offline
    eval. Never truncate to top 5 like collection_trace's UI preview does;
    rank-diagnosis needs enough headroom to see a doc drop out past rank 5."""
    return [
        {"rank": i + 1, "doc_id": row["doc_id"], "score": row["score"]}
        for i, row in enumerate(rows[:top_n])
    ]


def collection_ranked_trace(by_collection: dict[str, list[dict]], top_n: int = 10) -> dict:
    """Same as ranked_trace but per Milvus collection, for the dense/sparse spans."""
    return {
        name: {"hit_count": len(rows), "top_hits": ranked_trace(rows, top_n=top_n)}
        for name, rows in by_collection.items()
    }


def collection_trace(by_collection: dict[str, list[dict]]) -> dict:
    return {
        "collections": [
            {
                "name": name,
                "hit_count": len(rows),
                "top_hits": [
                    {
                        "chunk_id": row["chunk_id"],
                        "doc_id": row["doc_id"],
                        "score": row["score"],
                        "text_preview": row["text"][:200],
                    }
                    for row in rows[:5]
                ],
            }
            for name, rows in by_collection.items()
        ]
    }
