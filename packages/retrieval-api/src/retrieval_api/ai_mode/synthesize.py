from common.es_client import fetch_citations


async def synthesize(gateway, es_client, query: str, top_chunks: list[dict], citations: dict) -> dict:
    missing_doc_ids = [c["doc_id"] for c in top_chunks if c["doc_id"] not in citations]
    if missing_doc_ids:
        citations = {**citations, **await fetch_citations(es_client, missing_doc_ids)}

    chunk_block = "\n\n".join(f"[{c['doc_id']}] {c['text']}" for c in top_chunks)
    prompt = (
        f"Question: {query}\n\nRelevant excerpts:\n{chunk_block}\n\n"
        "Answer the question citing the doc_id in brackets for each claim."
    )
    answer = await gateway.chat(role="synthesis", messages=[{"role": "user", "content": prompt}])

    return {"answer": answer, "citations": citations}
