from fastapi import APIRouter, HTTPException

from common.config import get_settings
from common.es_client import get_es_client, fetch_fullcontent
from common.document_parser import parse_fullcontent, strip_tags_fallback

router = APIRouter()


@router.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    settings = get_settings()
    es_client = get_es_client(settings)
    try:
        fullcontent = await fetch_fullcontent(es_client, doc_id)
        if fullcontent is None:
            raise HTTPException(status_code=404, detail="Document not found")
        try:
            blocks = parse_fullcontent(fullcontent)
        except ValueError:
            # Some indexed documents have genuinely malformed fullcontent
            # XML at the source - degrade to plain text rather than 500.
            blocks = strip_tags_fallback(fullcontent)
        return {"doc_id": doc_id, "blocks": blocks}
    finally:
        await es_client.close()
