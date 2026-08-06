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
