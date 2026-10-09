"""OCR processing routes -- trigger OCR on pages/documents, fetch results."""

import os
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.auth import get_current_user
from app.config import settings
from app.database import get_db
from app.language_detection import build_ocr_language_annotation
from app.models import Document, OcrResult, Page, User, UserModel, VisualElement
from app.schemas import MessageResponse, OcrResultOut

router = APIRouter(prefix="/ocr", tags=["ocr"])


def _normalize_image_path(image_path: str | None) -> str | None:
    """Normalize mixed Windows/Unix separators to the current OS format."""
    if image_path is None:
        return None
    normalized = str(image_path).replace("\\", os.sep)
    normalized = os.path.normpath(normalized)
    return normalized


def _can_start_ocr(status: str | None) -> bool:
    """Return True when no OCR worker is currently processing the page.

    The explicit process endpoint clears prior OCR/visual results before a
    fresh run, so completed pages may be reprocessed. In-flight work is the
    one state that must never be enqueued a second time.
    """
    return status != "processing"


# ── OCR engine stub ──────────────────────────────────────────────────────────
# The actual OCR inference is implemented in a separate module.
# This function will be replaced once the model pipeline is ready.


async def _run_ocr_on_page(page_id: int, user_id: int, visual_mode: bool = False) -> None:
    """Run OCR inference on a single page and persist results.

    Fast mode detects text, text positions, and arrows only. Detailed visual
    mode additionally detects tables, boxes, circles, brackets, underlines,
    connectors, and diagrams.

    Uses Gemini Flash API when available (much better quality on camera photos).
    Falls back to TrOCR pipeline when no Gemini API key is configured.

    Pipeline:
    1. Auto-rotate: Gemini detects orientation; TrOCR tries all 4 rotations.
    2. Auto-crop: detect content bounds and store as page crop fields.
    3. Run OCR (with crop if set).
    4. Index results in Whoosh for full-text search.
    """
    import asyncio
    import json
    import logging

    from app.database import async_session  # local import to avoid circulars
    from app.ink_layers import InkLayerAnalyzer
    from app.ocr import (
        detect_content_bounds, get_engine, get_gemini_engine, get_openai_engine,
        has_gemini, has_openai, preprocess_image, should_try_trocr_fallback,
        detect_strike_through,
    )
    from app.routes.documents import _bake_rotation
    from app.routes.search import index_ocr_result

    logger = logging.getLogger(__name__)

    async with async_session() as db:
        # Mark page as processing.
        stmt = select(Page).where(Page.id == page_id)
        result = await db.execute(stmt)
        page = result.scalar_one_or_none()
        if page is None:
            return

        # A page may already be marked as "processing" by the request handler
        # immediately before this background task starts. Only reject a page
        # that is already completed; "processing" here means this worker owns it.
        if page.processing_status == "done":
            logger.warning("Skipping OCR run for already completed page %d", page_id)
            return

        page.image_path = _normalize_image_path(page.image_path) or page.image_path

        # Guard: verify image file exists before starting.
        import os
        if not os.path.exists(page.image_path):
            logger.error("Page %d image missing: %s", page_id, page.image_path)
            page.processing_status = "error"
            await db.commit()
            return

        page.processing_status = "processing"
        await db.flush()
        await db.commit()

        doc_stmt = select(Document).where(Document.id == page.document_id)
        doc_result = await db.execute(doc_stmt)
        document = doc_result.scalar_one_or_none()
        if document is None:
            return

        model_stmt = (
            select(UserModel)
            .where(UserModel.user_id == user_id)
            .order_by(UserModel.version.desc())
            .limit(1)
        )
        model_result = await db.execute(model_stmt)
        user_model = model_result.scalar_one_or_none()
        model_version = f"user-v{user_model.version}" if user_model else "base"

        segments: list | None = None
        visual_elements = []
        try:
            if has_openai():
                openai_engine = get_openai_engine()
                model_version = "openai-ocr"

                logger.info("Running OpenAI OCR on page %d", page_id)
                openai_result = await asyncio.to_thread(
                    openai_engine.process_page,
                    page.image_path, 0,
                )
                segments = openai_result.segments

            elif has_gemini():
                # ── Gemini Flash pipeline ──────────────────────────────
                gemini = get_gemini_engine()
                model_version = "gemini-flash"

                logger.info("Running Gemini OCR on page %d", page_id)

                # Keep the uploaded/display image immutable on the Gemini path.
                # Gemini is rotation-aware, so trusting one orientation guess and
                # baking it into the stored file is unsafe. A bad 90° guess made
                # the UI show a sideways image and could also make OCR return no text.
                # Perspective warp and deskew can similarly distort camera photos.
                # These transforms are therefore opt-in for controlled testing.
                if settings.GEMINI_AUTO_ROTATE:
                    logger.warning("GEMINI_AUTO_ROTATE enabled; stored image may be rotated.")
                    from app.ocr import preprocess_image as _pi
                    raw_img = await asyncio.to_thread(_pi, page.image_path, 0)
                    detected_rot = await asyncio.to_thread(gemini.detect_rotation, raw_img)
                    if detected_rot != 0:
                        new_path = _bake_rotation(page.image_path, detected_rot)
                        page.image_path = new_path
                        page.rotation = detected_rot
                        await db.flush()

                if settings.GEMINI_PAGE_WARP:
                    logger.warning("GEMINI_PAGE_WARP enabled; stored image may be warped.")
                    from app.ocr import preprocess_image as _pi, perspective_warp_page as _warp
                    if not page.page_warped:
                        warp_img = await asyncio.to_thread(_pi, page.image_path, 0)
                        corners = await asyncio.to_thread(gemini.detect_page_corners, warp_img)
                        if corners is not None:
                            new_path = await asyncio.to_thread(_warp, page.image_path, corners)
                            if new_path is not None:
                                page.image_path = new_path
                                page.crop_x = page.crop_y = None
                                page.crop_w = page.crop_h = None
                        page.page_warped = 1
                        await db.flush()

                if settings.GEMINI_DESKEW:
                    logger.warning("GEMINI_DESKEW enabled; stored image may be modified.")
                    from app.ocr import deskew_page as _deskew
                    deskewed = await asyncio.to_thread(_deskew, page.image_path)
                    if deskewed is not None:
                        page.image_path = deskewed
                        await db.flush()

                # Use the actual uploaded image for Gemini OCR. This keeps the
                # browser image and OCR coordinate system in the same pixel space.
                # Process with rotation=0 — file is already correctly
                # oriented (either originally or after baking above).
                gemini_result = await asyncio.to_thread(
                    gemini.process_page,
                    page.image_path, 0, visual_mode=visual_mode,
                )
                segments = gemini_result.segments
                visual_elements = gemini_result.visual_elements if visual_mode else [v for v in gemini_result.visual_elements if v.element_type == "arrow"]

            if segments is None:
                if not should_try_trocr_fallback():
                    logger.warning(
                        "Page %d: Gemini OCR is unavailable and TrOCR fallback is disabled; "
                        "keeping the page in an error state.",
                        page_id,
                    )
                    page.processing_status = "error"
                    await db.commit()
                    return

                logger.info("Gemini OCR unavailable for page %d; using TrOCR fallback", page_id)
                engine = get_engine()

                if user_model and user_model.lora_path:
                    engine.load_user_model(user_id, user_model.lora_path)
                else:
                    engine.unload_user_model()

                # Auto-rotation: try all 4 orientations.
                best_segments = None
                best_avg_conf = -1.0
                best_rotation = 0

                for candidate_rot in [0, 90, 180, 270]:
                    try:
                        segs = await asyncio.to_thread(
                            engine.process_page,
                            page.image_path, candidate_rot,
                        )
                    except Exception:
                        logger.warning(
                            "Auto-rotate: rotation=%d failed on page %d",
                            candidate_rot, page_id, exc_info=True,
                        )
                        continue

                    if not segs:
                        continue

                    avg_conf = sum(s.confidence for s in segs) / len(segs)
                    logger.info(
                        "Auto-rotate page %d: rotation=%d  avg_conf=%.4f  segments=%d",
                        page_id, candidate_rot, avg_conf, len(segs),
                    )
                    if avg_conf > best_avg_conf:
                        best_avg_conf = avg_conf
                        best_segments = segs
                        best_rotation = candidate_rot

                logger.info(
                    "Auto-rotate page %d: winner rotation=%d (conf=%.4f)",
                    page_id, best_rotation, best_avg_conf,
                )

                if best_rotation != 0:
                    new_path = _bake_rotation(page.image_path, best_rotation)
                    page.image_path = new_path
                    page.rotation = 0
                    await db.flush()

                # Auto-crop.
                image = preprocess_image(page.image_path, rotation=0)
                bounds = detect_content_bounds(image)
                if bounds is not None:
                    page.crop_x, page.crop_y, page.crop_w, page.crop_h = bounds
                else:
                    page.crop_x = page.crop_y = page.crop_w = page.crop_h = None
                await db.flush()

                # Final OCR with crop.
                crop = None
                if page.crop_x is not None:
                    crop = {
                        "x": page.crop_x, "y": page.crop_y,
                        "w": page.crop_w, "h": page.crop_h,
                    }

                segments = await asyncio.to_thread(
                    engine.process_page,
                    page.image_path, 0, crop,
                )
                visual_elements = []
        except Exception:
            logger.exception("Gemini OCR failed for page %d; falling back to TrOCR", page_id)
            segments = None

        if segments is None:
            if not should_try_trocr_fallback():
                logger.warning(
                    "Page %d: TrOCR fallback is disabled because the model is not cached locally. "
                    "Keeping the page in an error state instead of downloading the model.",
                    page_id,
                )
                page.processing_status = "error"
                await db.commit()
                return

            try:
                engine = get_engine()

                if user_model and user_model.lora_path:
                    engine.load_user_model(user_id, user_model.lora_path)
                else:
                    engine.unload_user_model()

                best_segments = None
                best_avg_conf = -1.0
                best_rotation = 0

                for candidate_rot in [0, 90, 180, 270]:
                    try:
                        segs = await asyncio.to_thread(
                            engine.process_page,
                            page.image_path, candidate_rot,
                        )
                    except Exception:
                        logger.warning(
                            "Auto-rotate: rotation=%d failed on page %d",
                            candidate_rot, page_id, exc_info=True,
                        )
                        continue

                    if not segs:
                        continue

                    avg_conf = sum(s.confidence for s in segs) / len(segs)
                    logger.info(
                        "Auto-rotate page %d: rotation=%d  avg_conf=%.4f  segments=%d",
                        page_id, candidate_rot, avg_conf, len(segs),
                    )
                    if avg_conf > best_avg_conf:
                        best_avg_conf = avg_conf
                        best_segments = segs
                        best_rotation = candidate_rot

                logger.info(
                    "Auto-rotate page %d: winner rotation=%d (conf=%.4f)",
                    page_id, best_rotation, best_avg_conf,
                )

                if best_rotation != 0:
                    new_path = _bake_rotation(page.image_path, best_rotation)
                    page.image_path = new_path
                    page.rotation = 0
                    await db.flush()

                image = preprocess_image(page.image_path, rotation=0)
                bounds = detect_content_bounds(image)
                if bounds is not None:
                    page.crop_x, page.crop_y, page.crop_w, page.crop_h = bounds
                else:
                    page.crop_x = page.crop_y = page.crop_w = page.crop_h = None
                await db.flush()

                crop = None
                if page.crop_x is not None:
                    crop = {
                        "x": page.crop_x, "y": page.crop_y,
                        "w": page.crop_w, "h": page.crop_h,
                    }

                segments = await asyncio.to_thread(
                    engine.process_page,
                    page.image_path, 0, crop,
                )
            except Exception:
                logger.exception("OCR processing failed for page %d", page_id)
                page.processing_status = "error"
                await db.commit()
                return

        try:
            # Re-processing replaces previously detected visual structures.
            await db.execute(
                VisualElement.__table__.delete().where(VisualElement.page_id == page.id)
            )
            page_image = preprocess_image(page.image_path, rotation=0)
            ink_regions = InkLayerAnalyzer().analyze_image(page_image)

            # Gemini often returns one box for an entire handwritten line.
            # Probe approximate word boxes so a strike through one word does not
            # hide unrelated words. Split only when a word-level stroke is found.
            import copy
            word_level_segments = []
            for segment in segments:
                words = (segment.text or "").split()
                x, y, w, h = segment.bbox
                if len(words) < 2 or w < 40:
                    word_level_segments.append(segment)
                    continue

                weights = [max(1, len(word)) for word in words]
                gap_weight = 0.7
                total_weight = sum(weights) + max(0, len(words) - 1) * gap_weight
                cursor = float(x)
                candidates = []
                for word, weight in zip(words, weights):
                    word_width = w * (weight / total_weight)
                    word_x = int(round(cursor))
                    word_right = int(round(cursor + word_width))
                    word_bbox = (word_x, y, max(1, word_right - word_x), h)
                    struck = detect_strike_through(page_image, word_bbox)
                    candidates.append((word, word_bbox, struck))
                    cursor += word_width + w * (gap_weight / total_weight)

                if any(item[2] for item in candidates):
                    for word, word_bbox, struck in candidates:
                        word_segment = copy.copy(segment)
                        word_segment.text = word
                        word_segment.bbox = word_bbox
                        word_segment.struck_through = struck
                        word_level_segments.append(word_segment)
                else:
                    word_level_segments.append(segment)

            for segment in word_level_segments:
                bbox_x, bbox_y, bbox_w, bbox_h = segment.bbox
                matched_region = None
                for region in ink_regions:
                    if region.bbox == (0, 0, page_image.width, page_image.height):
                        continue
                    rx, ry, rw, rh = region.bbox
                    overlap = max(0, min(bbox_x + bbox_w, rx + rw) - max(bbox_x, rx)) * max(0, min(bbox_y + bbox_h, ry + rh) - max(bbox_y, ry))
                    if overlap > 0:
                        matched_region = region
                        break

                label = matched_region.label if matched_region else "MAIN_INK"
                metadata = json.dumps(matched_region.to_dict(), ensure_ascii=False) if matched_region else None
                language_meta = build_ocr_language_annotation(segment.text)
                ocr_row = OcrResult(
                    page_id=page.id,
                    bbox_x=bbox_x,
                    bbox_y=bbox_y,
                    bbox_w=bbox_w,
                    bbox_h=bbox_h,
                    text=segment.text,
                    confidence=segment.confidence,
                    translated_text=segment.text,
                    translation_applied=False,
                    language=language_meta["language"],
                    language_name=language_meta["language_name"],
                    script=language_meta["script"],
                    language_confidence=language_meta["confidence"],
                    model_version=model_version,
                    ink_layer=label,
                    ink_metadata=metadata,
                    struck_through=bool(
                        getattr(segment, "struck_through", False)
                        or detect_strike_through(page_image, (bbox_x, bbox_y, bbox_w, bbox_h))
                    ),
                )
                db.add(ocr_row)
                await db.flush()

                try:
                    index_ocr_result(
                        ocr_result_id=ocr_row.id,
                        user_id=user_id,
                        page_id=page.id,
                        document_id=document.id,
                        text=segment.text,
                    )
                except Exception:
                    logger.warning(
                        "Failed to index OCR result %d", ocr_row.id, exc_info=True,
                    )

            # Persist non-text structures separately from OCR text.
            for element in visual_elements:
                db.add(
                    VisualElement(
                        page_id=page.id,
                        element_type=element.element_type,
                        bbox_x=element.bbox[0],
                        bbox_y=element.bbox[1],
                        bbox_w=element.bbox[2],
                        bbox_h=element.bbox[3],
                        confidence=element.confidence,
                        label=element.label,
                        geometry=element.geometry,
                        element_metadata=element.metadata,
                    )
                )

            page.processing_status = "done"
        except Exception:
            logger.exception("OCR processing failed for page %d", page_id)
            page.processing_status = "error"

        await db.commit()


# ── Helpers ───────────────────────────────────────────────────────────────────


async def _verify_page_ownership(
    page_id: int, user_id: int, db: AsyncSession
) -> Page:
    """Return the Page if it belongs to *user_id*, else raise 404."""
    stmt = (
        select(Page)
        .join(Document, Page.document_id == Document.id)
        .where(Page.id == page_id, Document.user_id == user_id)
    )
    result = await db.execute(stmt)
    page = result.scalar_one_or_none()
    if page is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Page not found")
    return page


async def _verify_document_ownership(
    document_id: int, user_id: int, db: AsyncSession
) -> Document:
    """Return the Document if it belongs to *user_id*, else raise 404."""
    stmt = (
        select(Document)
        .options(selectinload(Document.pages))
        .where(Document.id == document_id, Document.user_id == user_id)
    )
    result = await db.execute(stmt)
    doc = result.scalar_one_or_none()
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return doc


class OcrProcessRequest(BaseModel):
    visual_mode: bool = False


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post("/process/{page_id}", response_model=MessageResponse)
async def process_page(
    page_id: int,
    background_tasks: BackgroundTasks,
    body: OcrProcessRequest | None = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> MessageResponse:
    """Trigger OCR processing on a single page.

    Processing runs as a background task so the response returns immediately.
    """
    page = await _verify_page_ownership(page_id, current_user.id, db)

    if not _can_start_ocr(page.processing_status):
        return MessageResponse(message=f"Page {page.id} is already being processed or completed")

    # Clear any previous OCR results for this page so we get a fresh run.
    old_results_stmt = select(OcrResult).where(OcrResult.page_id == page.id)
    old_results = await db.execute(old_results_stmt)
    for old in old_results.scalars():
        await db.delete(old)
    old_visual_stmt = select(VisualElement).where(VisualElement.page_id == page.id)
    old_visuals = await db.execute(old_visual_stmt)
    for old in old_visuals.scalars():
        await db.delete(old)

    page.processing_status = "processing"
    await db.flush()

    # Persist the processing marker before scheduling the task so the
    # background worker cannot be started twice by overlapping requests.
    await db.commit()
    background_tasks.add_task(
        _run_ocr_on_page,
        page.id,
        current_user.id,
        bool(body.visual_mode) if body else False,
    )
    return MessageResponse(message=f"OCR processing started for page {page.id}")


@router.get("/results/{page_id}", response_model=List[OcrResultOut])
async def get_ocr_results(
    page_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list:
    """Return all OCR results for a page."""
    await _verify_page_ownership(page_id, current_user.id, db)

    stmt = (
        select(OcrResult)
        .where(OcrResult.page_id == page_id)
        .order_by(OcrResult.bbox_y, OcrResult.bbox_x)
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


class ProcessBboxRequest(BaseModel):
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int


@router.post("/process-bbox/{page_id}", response_model=OcrResultOut)
async def process_bbox(
    page_id: int,
    body: ProcessBboxRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> OcrResult:
    """Run OCR on a user-drawn bounding box region and return the result synchronously."""
    from app.ocr import get_engine, get_gemini_engine, has_gemini, preprocess_image
    from app.routes.search import index_ocr_result
    import logging

    logger = logging.getLogger(__name__)

    page = await _verify_page_ownership(page_id, current_user.id, db)

    # Find document for search indexing.
    doc_stmt = select(Document).where(Document.id == page.document_id)
    doc_result = await db.execute(doc_stmt)
    document = doc_result.scalar_one_or_none()

    # Load the image — file is already correctly oriented (rotation
    # was baked by auto-rotate), so use rotation=0 to avoid double-rotating.
    image = preprocess_image(page.image_path, rotation=0)
    if page.crop_x is not None and page.crop_y is not None and page.crop_w is not None and page.crop_h is not None:
        image = image.crop((page.crop_x, page.crop_y, page.crop_x + page.crop_w, page.crop_y + page.crop_h))
        crop_x_off = page.crop_x
        crop_y_off = page.crop_y
    else:
        crop_x_off = 0
        crop_y_off = 0

    # Crop to the user-drawn bbox.
    bx = body.bbox_x - crop_x_off
    by = body.bbox_y - crop_y_off
    bbox_img = image.crop((bx, by, bx + body.bbox_w, by + body.bbox_h))

    # Check user model.
    model_stmt = (
        select(UserModel)
        .where(UserModel.user_id == current_user.id)
        .order_by(UserModel.version.desc())
        .limit(1)
    )
    model_result = await db.execute(model_stmt)
    user_model = model_result.scalar_one_or_none()

    if has_gemini():
        gemini = get_gemini_engine()
        text, confidence = gemini.process_single(bbox_img)
        model_version = "gemini-flash"
    else:
        model_version = f"user-v{user_model.version}" if user_model else "base"
        engine = get_engine()
        if user_model and user_model.lora_path:
            engine.load_user_model(current_user.id, user_model.lora_path)
        else:
            engine.unload_user_model()
        text, confidence = engine.process_single(bbox_img)

    language_meta = build_ocr_language_annotation(text or "")
    translation_meta = identify_language_and_translate_to_english(text or "")
    page_image = preprocess_image(page.image_path, rotation=0)
    ocr_row = OcrResult(
        page_id=page.id,
        bbox_x=body.bbox_x,
        bbox_y=body.bbox_y,
        bbox_w=body.bbox_w,
        bbox_h=body.bbox_h,
        text=text or "",
        confidence=confidence,
        translated_text=text or "",
        translation_applied=False,
        language=language_meta["language"],
        language_name=language_meta["language_name"],
        script=language_meta["script"],
        language_confidence=language_meta["confidence"],
        model_version=model_version,
        struck_through=detect_strike_through(
            page_image,
            (body.bbox_x, body.bbox_y, body.bbox_w, body.bbox_h),
        ),
    )
    db.add(ocr_row)
    await db.flush()

    # Index in Whoosh.
    if document and text:
        try:
            index_ocr_result(
                ocr_result_id=ocr_row.id,
                user_id=current_user.id,
                page_id=page.id,
                document_id=document.id,
                text=text,
            )
        except Exception:
            logger.warning("Failed to index OCR result %d in search", ocr_row.id, exc_info=True)

    await db.commit()
    await db.refresh(ocr_row)
    return ocr_row


@router.post("/process-document/{document_id}", response_model=MessageResponse)
async def process_document(
    document_id: int,
    background_tasks: BackgroundTasks,
    body: OcrProcessRequest | None = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> MessageResponse:
    """Trigger OCR processing on every page in a document."""
    document = await _verify_document_ownership(document_id, current_user.id, db)

    if not document.pages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document has no pages",
        )

    for page in document.pages:
        if not _can_start_ocr(page.processing_status):
            continue

        # Clear old results.
        old_stmt = select(OcrResult).where(OcrResult.page_id == page.id)
        old_results = await db.execute(old_stmt)
        for old in old_results.scalars():
            await db.delete(old)
        old_visual_stmt = select(VisualElement).where(VisualElement.page_id == page.id)
        old_visuals = await db.execute(old_visual_stmt)
        for old in old_visuals.scalars():
            await db.delete(old)

        page.processing_status = "processing"
        background_tasks.add_task(
            _run_ocr_on_page,
            page.id,
            current_user.id,
            bool(body.visual_mode) if body else False,
        )

    # Persist processing markers before the background tasks start. Without
    # this commit, a second request can observe the old state and enqueue duplicates.
    await db.commit()

    return MessageResponse(
        message=f"OCR processing started for {len(document.pages)} page(s) in document '{document.name}'"
    )


class ProcessingStatusItem(BaseModel):
    page_id: int
    document_id: int
    document_name: str
    status: str


class ProcessingStatusResponse(BaseModel):
    pages: list[ProcessingStatusItem]


@router.get("/processing-status", response_model=ProcessingStatusResponse)
async def processing_status(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ProcessingStatusResponse:
    """Return any pages currently being processed or recently completed for the user.

    Frontend polls this to show notifications when background OCR finishes.
    """
    stmt = (
        select(Page, Document.id, Document.name)
        .join(Document, Page.document_id == Document.id)
        .where(
            Document.user_id == current_user.id,
            Page.processing_status.in_(["processing", "done", "error"]),
        )
    )
    result = await db.execute(stmt)
    rows = result.all()

    items = [
        ProcessingStatusItem(
            page_id=page.id,
            document_id=doc_id,
            document_name=doc_name,
            status=page.processing_status or "idle",
        )
        for page, doc_id, doc_name in rows
    ]

    # Do not mutate terminal state while reporting it. The old read-to-clear
    # behavior meant two tabs could race: one tab consumed `done`/`error`, the
    # other saw nothing and kept polling. Terminal states are now cleared only
    # when a fresh OCR request explicitly starts that page again.
    return ProcessingStatusResponse(pages=items)


# ── Training mode ─────────────────────────────────────────────────────────────


class UpdateBboxRequest(BaseModel):
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int


@router.put("/result/{result_id}/bbox", response_model=OcrResultOut)
async def update_result_bbox(
    result_id: int,
    body: UpdateBboxRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> OcrResult:
    """Update the bounding box of an existing OCR result (training mode)."""
    stmt = (
        select(OcrResult)
        .join(Page, OcrResult.page_id == Page.id)
        .join(Document, Page.document_id == Document.id)
        .where(OcrResult.id == result_id, Document.user_id == current_user.id)
    )
    result = await db.execute(stmt)
    ocr_row = result.scalar_one_or_none()
    if ocr_row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Result not found")

    ocr_row.bbox_x = body.bbox_x
    ocr_row.bbox_y = body.bbox_y
    ocr_row.bbox_w = body.bbox_w
    ocr_row.bbox_h = body.bbox_h
    await db.flush()
    return ocr_row
