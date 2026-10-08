"""Correction routes -- submit corrections, list them, and the "Play" game."""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.auth import get_current_user
from app.database import get_db
from app.models import Correction, Document, OcrResult, Page, User
from app.schemas import (
    CorrectionCreate,
    CorrectionOut,
    PlayBatchResponse,
    PlayItem,
    PlaySubmit,
)

router = APIRouter(prefix="/corrections", tags=["corrections"])


# ── Helpers ───────────────────────────────────────────────────────────────────


async def _verify_ocr_result_ownership(
    ocr_result_id: int, user_id: int, db: AsyncSession
) -> OcrResult:
    """Return the OcrResult if it ultimately belongs to *user_id*."""
    stmt = (
        select(OcrResult)
        .join(Page, OcrResult.page_id == Page.id)
        .join(Document, Page.document_id == Document.id)
        .where(OcrResult.id == ocr_result_id, Document.user_id == user_id)
    )
    result = await db.execute(stmt)
    ocr = result.scalar_one_or_none()
    if ocr is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="OCR result not found",
        )
    return ocr


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post("", response_model=CorrectionOut, status_code=status.HTTP_201_CREATED)
async def create_correction(
    body: CorrectionCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Correction:
    """Submit a correction for an OCR result."""
    ocr_result = await _verify_ocr_result_ownership(body.ocr_result_id, current_user.id, db)

    correction = Correction(
        ocr_result_id=ocr_result.id,
        original_text=ocr_result.text,
        corrected_text=body.corrected_text,
    )
    db.add(correction)

    # Update the displayed text so the UI reflects the correction.
    ocr_result.text = body.corrected_text
    await db.flush()
    return correction


@router.get("", response_model=List[CorrectionOut])
async def list_corrections(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list:
    """List corrections made by the current user, newest first."""
    stmt = (
        select(Correction)
        .join(OcrResult, Correction.ocr_result_id == OcrResult.id)
        .join(Page, OcrResult.page_id == Page.id)
        .join(Document, Page.document_id == Document.id)
        .where(Document.user_id == current_user.id)
        .order_by(Correction.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


@router.get("/play", response_model=PlayBatchResponse)
async def play_batch(
    batch_size: int = Query(default=10, ge=1, le=50),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> PlayBatchResponse:
    """Get the next batch of OCR results for the correction game.

    Prioritisation:
    1. Lowest confidence (model is most uncertain).
    2. Among equal-confidence items, prefer those whose page or document
       already has corrections (similar handwriting patterns).
    Items that have already been corrected are excluded.
    """
    # Sub-query: OCR result IDs that already have at least one correction.
    corrected_ids_sq = select(Correction.ocr_result_id).distinct().subquery()

    # Sub-query: count of corrections per document (signals "similar
    # handwriting has been corrected before").
    doc_correction_count_sq = (
        select(
            Document.id.label("doc_id"),
            func.count(Correction.id).label("corr_count"),
        )
        .join(Page, Document.id == Page.document_id)
        .join(OcrResult, Page.id == OcrResult.page_id)
        .join(Correction, OcrResult.id == Correction.ocr_result_id)
        .group_by(Document.id)
        .subquery()
    )

    # Main query: uncorrected OCR results for this user, ordered by priority.
    stmt = (
        select(OcrResult, Page.image_path, Page.rotation, Document.name.label("document_name"))
        .join(Page, OcrResult.page_id == Page.id)
        .join(Document, Page.document_id == Document.id)
        .outerjoin(doc_correction_count_sq, Document.id == doc_correction_count_sq.c.doc_id)
        .where(
            Document.user_id == current_user.id,
            OcrResult.id.notin_(select(Correction.ocr_result_id).distinct()),
        )
        .order_by(
            OcrResult.confidence.asc(),  # lowest confidence first
            func.coalesce(doc_correction_count_sq.c.corr_count, 0).desc(),  # docs with more corrections first
        )
        .limit(batch_size)
    )

    result = await db.execute(stmt)
    rows = result.all()

    items = [
        PlayItem(
            ocr_result_id=ocr.id,
            page_id=ocr.page_id,
            page_image_path=image_path,
            bbox_x=ocr.bbox_x,
            bbox_y=ocr.bbox_y,
            bbox_w=ocr.bbox_w,
            bbox_h=ocr.bbox_h,
            recognized_text=ocr.text,
            confidence=ocr.confidence,
            document_name=doc_name,
            page_rotation=page_rotation,
        )
        for ocr, image_path, page_rotation, doc_name in rows
    ]

    # Count remaining uncorrected items.
    remaining_stmt = (
        select(func.count(OcrResult.id))
        .join(Page, OcrResult.page_id == Page.id)
        .join(Document, Page.document_id == Document.id)
        .where(
            Document.user_id == current_user.id,
            OcrResult.id.notin_(select(Correction.ocr_result_id).distinct()),
        )
    )
    remaining_result = await db.execute(remaining_stmt)
    total_remaining: int = remaining_result.scalar_one()

    return PlayBatchResponse(
        items=items,
        remaining=max(0, total_remaining - len(items)),
    )


@router.post("/play/submit", response_model=CorrectionOut, status_code=status.HTTP_201_CREATED)
async def play_submit(
    body: PlaySubmit,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Correction:
    """Submit a correction from the Play game mode."""
    ocr_result = await _verify_ocr_result_ownership(body.ocr_result_id, current_user.id, db)

    correction = Correction(
        ocr_result_id=ocr_result.id,
        original_text=ocr_result.text,
        corrected_text=body.corrected_text,
        corrected_bbox_x=body.corrected_bbox_x,
        corrected_bbox_y=body.corrected_bbox_y,
        corrected_bbox_w=body.corrected_bbox_w,
        corrected_bbox_h=body.corrected_bbox_h,
    )
    db.add(correction)

    # Update the displayed text so the UI reflects the correction.
    ocr_result.text = body.corrected_text
    await db.flush()
    return correction
