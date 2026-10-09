"""Full-text search across OCR results using Whoosh."""

import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from whoosh import index as whoosh_index
from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import ID, TEXT, Schema
from whoosh.qparser import MultifieldParser, OrGroup
from whoosh import query as whoosh_query

from app.auth import get_current_user
from app.database import get_db
from app.models import Document, OcrResult, Page, User
from app.schemas import SearchResponse, SearchResult

router = APIRouter(prefix="/search", tags=["search"])

# Keep the index anchored to the backend working directory, not whichever
# directory happens to launch Uvicorn (which previously created multiple empty
# indexes when starting the app from different folders).
_BACKEND_ROOT = Path(__file__).resolve().parents[2]
WHOOSH_DIR = str(_BACKEND_ROOT / "data" / "whoosh_index")

_schema = Schema(
    ocr_result_id=ID(stored=True, unique=True),
    user_id=ID(stored=True),
    page_id=ID(stored=True),
    document_id=ID(stored=True),
    text=TEXT(analyzer=StemmingAnalyzer(), stored=True),
)


def _get_or_create_index() -> whoosh_index.Index:
    """Return the Whoosh index, creating it on disk if necessary."""
    os.makedirs(WHOOSH_DIR, exist_ok=True)
    if whoosh_index.exists_in(WHOOSH_DIR):
        return whoosh_index.open_dir(WHOOSH_DIR)
    return whoosh_index.create_in(WHOOSH_DIR, _schema)


def get_search_index() -> whoosh_index.Index:
    """Public accessor for the search index."""
    return _get_or_create_index()


def index_ocr_result(
    ocr_result_id: int,
    user_id: int,
    page_id: int,
    document_id: int,
    text: str,
) -> None:
    """Add or update one OCR result in the full-text index."""
    clean_text = (text or "").strip()
    if not clean_text:
        return
    ix = get_search_index()
    writer = ix.writer()
    try:
        writer.update_document(
            ocr_result_id=str(ocr_result_id),
            user_id=str(user_id),
            page_id=str(page_id),
            document_id=str(document_id),
            text=clean_text,
        )
        writer.commit()
    except Exception:
        writer.cancel()
        raise


def remove_document_from_index(document_id: int) -> None:
    """Remove all indexed entries for a document."""
    ix = get_search_index()
    writer = ix.writer()
    writer.delete_by_term("document_id", str(document_id))
    writer.commit()


async def rebuild_index_for_user(user_id: int, db: AsyncSession) -> int:
    """Rebuild this user's index records from the source database."""
    stmt = (
        select(OcrResult, Page.document_id)
        .join(Page, OcrResult.page_id == Page.id)
        .join(Document, Page.document_id == Document.id)
        .where(Document.user_id == user_id)
    )
    rows = (await db.execute(stmt)).all()
    ix = get_search_index()
    writer = ix.writer()
    try:
        writer.delete_by_term("user_id", str(user_id))
        count = 0
        for ocr, doc_id in rows:
            text = (ocr.text or "").strip()
            if not text:
                continue
            writer.update_document(
                ocr_result_id=str(ocr.id),
                user_id=str(user_id),
                page_id=str(ocr.page_id),
                document_id=str(doc_id),
                text=text,
            )
            count += 1
        writer.commit()
        return count
    except Exception:
        writer.cancel()
        raise


@router.get("", response_model=SearchResponse)
async def search(
    q: str = Query(..., min_length=1, max_length=500, description="Search query"),
    limit: int = Query(default=30, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SearchResponse:
    """Search the current user's OCR text and return navigable page metadata.

    The Whoosh index is a cache, so rebuild it for this user when empty or stale.
    This means older documents, documents indexed before an app restart, and
    corrected OCR text remain searchable without asking users to re-upload.
    """
    query_text = q.strip()
    if not query_text:
        return SearchResponse(query=q, total=0, results=[])

    ix = get_search_index()
    parser = MultifieldParser(["text"], schema=ix.schema, group=OrGroup)

    def parse_query(index: whoosh_index.Index, value: str):
        # Bind the parser to the opened index's schema. The on-disk index is
        # opened before query parsing, including after a rebuild.
        active_parser = MultifieldParser(["text"], schema=index.schema, group=OrGroup)
        try:
            return active_parser.parse(value)
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid search query") from exc

    def collect_ids(index: whoosh_index.Index, parsed_query) -> list[int]:
        with index.searcher() as searcher:
            hits = searcher.search(
                parsed_query,
                filter=whoosh_query.Term("user_id", str(current_user.id)),
                limit=limit,
            )
            return [int(hit["ocr_result_id"]) for hit in hits]

    # Avoid rebuilding on every keystroke. If this query has no indexed match,
    # rebuild this user's index from the database and retry. This repairs empty
    # or stale indexes without making ordinary searches perform a full DB scan.
    parsed_query = parse_query(ix, query_text)
    matching_ids = collect_ids(ix, parsed_query)
    if not matching_ids:
        await rebuild_index_for_user(current_user.id, db)
        ix = get_search_index()
        parsed_query = parse_query(ix, query_text)
        matching_ids = collect_ids(ix, parsed_query)

    if not matching_ids and len(query_text) <= 3:
        # Permit short words/acronyms (e.g. OCR, AI, F1) to match within a
        # token. Wildcard queries operate on indexed terms, not full stored text.
        with ix.searcher() as searcher:
            term_hits = searcher.search(
                whoosh_query.And([
                    whoosh_query.Term("user_id", str(current_user.id)),
                    whoosh_query.Wildcard("text", f"*{query_text.lower()}*"),
                ]),
                limit=limit,
            )
            matching_ids = [int(hit["ocr_result_id"]) for hit in term_hits]

    if not matching_ids:
        return SearchResponse(query=query_text, total=0, results=[])

    stmt = (
        select(
            OcrResult,
            Page.image_path,
            Page.page_number,
            Document.id.label("doc_id"),
            Document.name.label("doc_name"),
        )
        .join(Page, OcrResult.page_id == Page.id)
        .join(Document, Page.document_id == Document.id)
        .where(
            OcrResult.id.in_(matching_ids),
            Document.user_id == current_user.id,
        )
    )
    rows = (await db.execute(stmt)).all()
    id_order = {ocr_id: index for index, ocr_id in enumerate(matching_ids)}
    rows_sorted = sorted(rows, key=lambda row: id_order.get(row[0].id, 9999))

    search_results = [
        SearchResult(
            ocr_result_id=ocr.id,
            page_id=ocr.page_id,
            page_image_path=image_path,
            document_id=doc_id,
            document_name=doc_name,
            text=ocr.text,
            confidence=ocr.confidence,
            bbox_x=ocr.bbox_x,
            bbox_y=ocr.bbox_y,
            bbox_w=ocr.bbox_w,
            bbox_h=ocr.bbox_h,
            page_number=page_number,
        )
        for ocr, image_path, page_number, doc_id, doc_name in rows_sorted
    ]
    return SearchResponse(query=query_text, total=len(search_results), results=search_results)
