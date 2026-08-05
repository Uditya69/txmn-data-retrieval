from fastapi import APIRouter, HTTPException

from common.config import get_settings
from common.es_client import get_es_client, fetch_fullcontent
from common.document_parser import parse_fullcontent

router = APIRouter()


@router.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    settings = get_settings()
    es_client = get_es_client(settings)
    try:
        fullcontent = await fetch_fullcontent(es_client, doc_id)
        if fullcontent is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"doc_id": doc_id, "blocks": parse_fullcontent(fullcontent)}
    finally:
        await es_client.close()
