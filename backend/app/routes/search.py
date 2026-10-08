"""Full-text search across OCR results using Whoosh."""

import os
import shutil
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from whoosh import index as whoosh_index
from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import ID, NUMERIC, TEXT, Schema
from whoosh.qparser import MultifieldParser, OrGroup

from app.auth import get_current_user
from app.database import get_db
from app.models import Document, OcrResult, Page, User
from app.schemas import SearchResponse, SearchResult

router = APIRouter(prefix="/search", tags=["search"])

# ── Whoosh schema & index management ─────────────────────────────────────────

WHOOSH_DIR = os.path.join("data", "whoosh_index")

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
    """Public accessor for the singleton Whoosh index."""
    return _get_or_create_index()


# ── Index maintenance helpers (called from other modules) ─────────────────────


def index_ocr_result(
    ocr_result_id: int,
    user_id: int,
    page_id: int,
    document_id: int,
    text: str,
) -> None:
    """Add or update a single OCR result in the search index."""
    ix = get_search_index()
    writer = ix.writer()
    writer.update_document(
        ocr_result_id=str(ocr_result_id),
        user_id=str(user_id),
        page_id=str(page_id),
        document_id=str(document_id),
        text=text,
    )
    writer.commit()


def remove_document_from_index(document_id: int) -> None:
    """Remove all indexed entries for a document."""
    ix = get_search_index()
    writer = ix.writer()
    writer.delete_by_term("document_id", str(document_id))
    writer.commit()


async def rebuild_index_for_user(user_id: int, db: AsyncSession) -> int:
    """(Re)build the entire search index for a user. Returns count indexed."""
    stmt = (
        select(OcrResult, Page.document_id)
        .join(Page, OcrResult.page_id == Page.id)
        .join(Document, Page.document_id == Document.id)
        .where(Document.user_id == user_id)
    )
    result = await db.execute(stmt)
    rows = result.all()

    ix = get_search_index()
    writer = ix.writer()

    # Remove old entries for this user first.
    writer.delete_by_term("user_id", str(user_id))

    count = 0
    for ocr, doc_id in rows:
        writer.add_document(
            ocr_result_id=str(ocr.id),
            user_id=str(user_id),
            page_id=str(ocr.page_id),
            document_id=str(doc_id),
            text=ocr.text,
        )
        count += 1

    writer.commit()
    return count


# ── Endpoint ──────────────────────────────────────────────────────────────────


@router.get("", response_model=SearchResponse)
async def search(
    q: str = Query(..., min_length=1, max_length=500, description="Search query"),
    limit: int = Query(default=30, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SearchResponse:
    """Full-text search across the current user's OCR results.

    Returns matching OCR segments with their page image path, bounding box,
    and parent document info so the frontend can render highlighted results.
    """
    ix = get_search_index()
    parser = MultifieldParser(["text"], schema=ix.schema, group=OrGroup)
    query = parser.parse(q)

    matching_ids: list[int] = []

    with ix.searcher() as searcher:
        results = searcher.search(
            query,
            filter=whoosh_index.query.Term("user_id", str(current_user.id))
            if hasattr(whoosh_index, "query")
            else None,
            limit=limit * 3,  # over-fetch since we filter by user below
        )

        for hit in results:
            if hit["user_id"] == str(current_user.id):
                matching_ids.append(int(hit["ocr_result_id"]))
                if len(matching_ids) >= limit:
                    break

    if not matching_ids:
        return SearchResponse(query=q, total=0, results=[])

    # Hydrate from the database to get full info.
    stmt = (
        select(OcrResult, Page.image_path, Document.id.label("doc_id"), Document.name.label("doc_name"))
        .join(Page, OcrResult.page_id == Page.id)
        .join(Document, Page.document_id == Document.id)
        .where(
            OcrResult.id.in_(matching_ids),
            Document.user_id == current_user.id,
        )
    )
    result = await db.execute(stmt)
    rows = result.all()

    # Preserve the Whoosh relevance ordering.
    id_order = {oid: idx for idx, oid in enumerate(matching_ids)}
    rows_sorted = sorted(rows, key=lambda r: id_order.get(r[0].id, 9999))

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
        )
        for ocr, image_path, doc_id, doc_name in rows_sorted
    ]

    return SearchResponse(query=q, total=len(search_results), results=search_results)
