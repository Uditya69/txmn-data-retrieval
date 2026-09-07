from fastapi import APIRouter, HTTPException

from common.config import get_settings
from common.es_client import get_es_client, fetch_fullcontent, fetch_highlighted_fullcontent, fetch_document_metadata
from common.document_parser import parse_fullcontent, strip_tags_fallback

router = APIRouter()


@router.get("/documents/{doc_id}")
async def get_document(doc_id: str, query: str | None = None):
    settings = get_settings()
    es_client = get_es_client(settings)
    try:
        # `query` is the search the reader was opened from (prod parity -
        # real server-side highlighting over the whole document, see
        # fetch_highlighted_fullcontent's own docstring) - absent for a
        # bookmarked/direct link with no search context, where plain
        # unhighlighted fullcontent is the only sensible fallback.
        if query:
            fullcontent = await fetch_highlighted_fullcontent(es_client, doc_id, query)
        else:
            fullcontent = await fetch_fullcontent(es_client, doc_id)
        if fullcontent is None:
            raise HTTPException(status_code=404, detail="Document not found")
        try:
            blocks = parse_fullcontent(fullcontent)
        except ValueError:
            # Some indexed documents have genuinely malformed fullcontent XML
            # at the source - degrade to plain text rather than 500. A
            # highlighted phrase match straddling an element boundary can
            # (rarely) produce the same kind of malformed XML - same
            # degradation, not a separate failure mode.
            blocks = strip_tags_fallback(fullcontent)
        metadata = await fetch_document_metadata(es_client, doc_id) or {}
        return {
            "doc_id": doc_id,
            "heading": metadata.get("heading"),
            "subheading": metadata.get("subheading"),
            "year": metadata.get("year"),
            "blocks": blocks,
        }
    finally:
        await es_client.close()
