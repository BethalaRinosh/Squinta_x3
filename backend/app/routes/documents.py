"""Document management routes -- upload, camera capture, list, detail, delete."""

import base64
import os
import uuid
from typing import List

import aiofiles
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.auth import get_current_user
from app.config import settings
from app.database import get_db
from app.models import Document, Page, User
from app.schemas import DocumentListItem, DocumentOut, MessageResponse, PageOut

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp", ".gif", ".pdf", ".heic", ".heif"}


def _user_upload_dir(user_id: int) -> str:
    """Return (and create if needed) the per-user upload directory."""
    path = os.path.join(settings.UPLOAD_DIR, str(user_id))
    os.makedirs(path, exist_ok=True)
    return path


def _validate_image_extension(filename: str) -> str:
    """Return the lowered extension or raise 400."""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )
    return ext


async def _save_upload_file(upload: UploadFile, dest: str) -> None:
    """Stream an UploadFile to disk."""
    async with aiofiles.open(dest, "wb") as f:
        while chunk := await upload.read(1024 * 256):  # 256 KiB
            await f.write(chunk)


def _bake_exif_rotation(path: str) -> None:
    """Apply EXIF orientation to pixel data and save back, removing the EXIF tag."""
    from PIL import Image, ImageOps
    try:
        with Image.open(path) as img:
            transposed = ImageOps.exif_transpose(img)
            if transposed is not img:
                transposed.save(path)
    except Exception:
        pass  # best-effort; not all formats have EXIF


def _convert_heic_to_jpeg(src_path: str) -> str:
    """Convert a HEIC/HEIF file to JPEG. Returns the new path."""
    from pillow_heif import register_heif_opener
    from PIL import Image, ImageOps
    register_heif_opener()

    jpeg_path = os.path.splitext(src_path)[0] + ".jpg"
    with Image.open(src_path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        img.save(jpeg_path, "JPEG", quality=92)

    # Remove the original HEIC
    os.remove(src_path)
    return jpeg_path


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get("", response_model=List[DocumentListItem])
async def list_documents(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list:
    """List all documents belonging to the current user."""
    from app.models import OcrResult

    # Sub-query to count pages per document.
    page_count_sq = (
        select(Page.document_id, func.count(Page.id).label("page_count"))
        .group_by(Page.document_id)
        .subquery()
    )

    # Sub-query to count OCR results per document.
    ocr_count_sq = (
        select(Page.document_id, func.count(OcrResult.id).label("ocr_count"))
        .join(OcrResult, Page.id == OcrResult.page_id)
        .group_by(Page.document_id)
        .subquery()
    )

    stmt = (
        select(
            Document.id,
            Document.user_id,
            Document.name,
            Document.source,
            Document.created_at,
            func.coalesce(page_count_sq.c.page_count, 0).label("page_count"),
            func.coalesce(ocr_count_sq.c.ocr_count, 0).label("ocr_result_count"),
        )
        .outerjoin(page_count_sq, Document.id == page_count_sq.c.document_id)
        .outerjoin(ocr_count_sq, Document.id == ocr_count_sq.c.document_id)
        .where(Document.user_id == current_user.id)
        .order_by(Document.created_at.desc())
    )

    result = await db.execute(stmt)
    rows = result.all()

    return [
        DocumentListItem(
            id=row.id,
            user_id=row.user_id,
            name=row.name,
            source=row.source,
            created_at=row.created_at,
            page_count=row.page_count,
            ocr_result_count=row.ocr_result_count,
        )
        for row in rows
    ]


@router.post("/upload", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
async def upload_document(
    files: List[UploadFile] = File(...),
    name: str = Form(default=""),
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Document:
    """Upload one or more image files and create a Document with Pages."""
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files provided")

    import logging
    logger = logging.getLogger(__name__)
    for f in files:
        logger.warning(f"Upload received: filename={f.filename!r}, content_type={f.content_type!r}")

    # Default name = first filename without extension.
    doc_name = name.strip() or os.path.splitext(files[0].filename or "untitled")[0]

    document = Document(user_id=current_user.id, name=doc_name, source="upload")
    db.add(document)
    await db.flush()  # get document.id

    upload_dir = _user_upload_dir(current_user.id)

    for idx, upload_file in enumerate(files):
        ext = _validate_image_extension(upload_file.filename or "image.png")
        unique_name = f"{uuid.uuid4().hex}{ext}"
        dest = os.path.join(upload_dir, unique_name)
        await _save_upload_file(upload_file, dest)

        # Convert HEIC/HEIF to JPEG so browsers can display them
        if ext in (".heic", ".heif"):
            dest = _convert_heic_to_jpeg(dest)
        else:
            _bake_exif_rotation(dest)

        page = Page(
            document_id=document.id,
            image_path=dest,
            page_number=idx + 1,
        )
        db.add(page)

    # Commit the document, pages, and processing markers before the
    # background worker opens its own database session. Without this commit,
    # the worker can race the request transaction and observe stale state.
    await db.commit()

    # Auto-trigger OCR on all pages.
    from app.routes.ocr import _run_ocr_on_page
    re_stmt = (
        select(Document)
        .options(selectinload(Document.pages).selectinload(Page.ocr_results))
        .where(Document.id == document.id)
    )
    re_result = await db.execute(re_stmt)
    doc = re_result.scalar_one()
    for pg in doc.pages:
        pg.processing_status = "processing"
        background_tasks.add_task(_run_ocr_on_page, pg.id, current_user.id)
    await db.commit()

    # Re-read the document after committing so the response contains the
    # committed page state and fresh image paths.
    re_result = await db.execute(re_stmt)
    doc = re_result.scalar_one()
    return doc


class CameraCaptureRequest(BaseModel):
    image: str  # base64-encoded image (data URI or raw base64)
    name: str = "Camera capture"


@router.post("/camera", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
async def camera_capture(
    body: CameraCaptureRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Document:
    """Accept a base64-encoded image captured from the device camera."""
    image_data = body.image
    name = body.name
    # Strip optional data-URI prefix: "data:image/png;base64,..."
    raw = image_data
    if "," in raw:
        header, raw = raw.split(",", 1)
    else:
        header = ""

    try:
        img_bytes = base64.b64decode(raw)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid base64 image data",
        ) from exc

    # Guess extension from data-URI header, default to .png.
    ext = ".png"
    if "jpeg" in header or "jpg" in header:
        ext = ".jpg"
    elif "webp" in header:
        ext = ".webp"

    upload_dir = _user_upload_dir(current_user.id)
    unique_name = f"{uuid.uuid4().hex}{ext}"
    dest = os.path.join(upload_dir, unique_name)

    async with aiofiles.open(dest, "wb") as f:
        await f.write(img_bytes)

    _bake_exif_rotation(dest)

    document = Document(user_id=current_user.id, name=name.strip() or "Camera capture", source="camera")
    db.add(document)
    await db.flush()

    page = Page(document_id=document.id, image_path=dest, page_number=1)
    db.add(page)
    await db.flush()

    # Auto-trigger OCR.
    from app.routes.ocr import _run_ocr_on_page
    page.processing_status = "processing"
    background_tasks.add_task(_run_ocr_on_page, page.id, current_user.id)
    await db.flush()

    stmt = (
        select(Document)
        .options(selectinload(Document.pages).selectinload(Page.ocr_results))
        .where(Document.id == document.id)
    )
    result = await db.execute(stmt)
    return result.scalar_one()


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Document:
    """Return a document with its pages and their OCR results."""
    from app.models import OcrResult  # local to avoid circular at module level

    stmt = (
        select(Document)
        .options(
            selectinload(Document.pages).selectinload(Page.ocr_results),
        )
        .where(Document.id == document_id, Document.user_id == current_user.id)
    )
    result = await db.execute(stmt)
    document = result.scalar_one_or_none()

    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    return document


@router.delete("/{document_id}", response_model=MessageResponse)
async def delete_document(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> MessageResponse:
    """Delete a document and its associated pages / OCR results.

    Image files are also removed from disk.
    """
    from app.models import OcrResult, Correction

    stmt = (
        select(Document)
        .options(
            selectinload(Document.pages)
            .selectinload(Page.ocr_results)
            .selectinload(OcrResult.corrections),
        )
        .where(Document.id == document_id, Document.user_id == current_user.id)
    )
    result = await db.execute(stmt)
    document = result.scalar_one_or_none()

    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    # Clean up image files.
    for page in document.pages:
        try:
            if os.path.isfile(page.image_path):
                os.remove(page.image_path)
        except OSError:
            pass  # best-effort deletion

    await db.delete(document)
    return MessageResponse(message="Document deleted")


class RotateRequest(BaseModel):
    rotation: int  # 0, 90, 180, 270


def _bake_rotation(image_path: str, rotation: int) -> str:
    """Physically rotate the image file on disk so it's always correctly oriented.

    Returns the (possibly new) image path.  The file is saved with a fresh
    UUID name so that the browser URL changes and caches are busted.
    """
    if rotation == 0:
        return os.path.normpath(str(image_path).replace("\\", os.sep))
    image_path = os.path.normpath(str(image_path).replace("\\", os.sep))
    from PIL import Image

    img = Image.open(image_path)
    img.load()  # read all pixel data into memory
    if img.mode != "RGB":
        img = img.convert("RGB")
    # PIL rotate is counter-clockwise; user rotation is clockwise
    rotated = img.rotate(-rotation, expand=True)
    img.close()

    # Save to a new filename so the URL changes (busts browser cache).
    directory = os.path.dirname(image_path)
    ext = os.path.splitext(image_path)[1] or ".jpg"
    new_path = os.path.normpath(os.path.join(directory, f"{uuid.uuid4().hex}{ext}"))
    rotated.save(new_path)

    # Remove old file.
    try:
        os.remove(image_path)
    except OSError:
        pass

    return new_path


@router.post("/pages/{page_id}/rotate", response_model=PageOut)
async def rotate_page(
    page_id: int,
    body: RotateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Page:
    """Rotate a page image by the given amount and bake into the file.

    The rotation is physically applied to the image on disk so that
    ``page.rotation`` is always 0 after this call.  Crop and OCR results
    are cleared since they relate to the previous orientation.
    """
    if body.rotation not in (0, 90, 180, 270):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="rotation must be 0, 90, 180, or 270",
        )

    stmt = (
        select(Page)
        .join(Document, Page.document_id == Document.id)
        .options(selectinload(Page.ocr_results))
        .where(Page.id == page_id, Document.user_id == current_user.id)
    )
    result = await db.execute(stmt)
    page = result.scalar_one_or_none()
    if page is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Page not found")

    # Physically rotate the image file and keep rotation=0 in DB.
    new_path = _bake_rotation(page.image_path, body.rotation)
    page.image_path = new_path
    page.rotation = 0
    page.page_warped = 0  # re-detect page corners on next OCR run

    # Clear crop (coords were relative to old orientation).
    page.crop_x = None
    page.crop_y = None
    page.crop_w = None
    page.crop_h = None

    # Clear old OCR results (bbox coords are for old orientation).
    from app.models import OcrResult
    old_stmt = select(OcrResult).where(OcrResult.page_id == page.id)
    old_results = await db.execute(old_stmt)
    for old in old_results.scalars():
        await db.delete(old)

    await db.flush()
    return page


class CropRequest(BaseModel):
    crop_x: int
    crop_y: int
    crop_w: int
    crop_h: int


@router.post("/pages/{page_id}/crop", response_model=PageOut)
async def set_page_crop(
    page_id: int,
    body: CropRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Page:
    """Set a crop region for a page."""
    stmt = (
        select(Page)
        .join(Document, Page.document_id == Document.id)
        .options(selectinload(Page.ocr_results))
        .where(Page.id == page_id, Document.user_id == current_user.id)
    )
    result = await db.execute(stmt)
    page = result.scalar_one_or_none()
    if page is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Page not found")

    page.crop_x = body.crop_x
    page.crop_y = body.crop_y
    page.crop_w = body.crop_w
    page.crop_h = body.crop_h
    await db.flush()
    return page


@router.post("/pages/{page_id}/crop/clear", response_model=PageOut)
async def clear_page_crop(
    page_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Page:
    """Clear the crop region for a page."""
    stmt = (
        select(Page)
        .join(Document, Page.document_id == Document.id)
        .options(selectinload(Page.ocr_results))
        .where(Page.id == page_id, Document.user_id == current_user.id)
    )
    result = await db.execute(stmt)
    page = result.scalar_one_or_none()
    if page is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Page not found")

    page.crop_x = None
    page.crop_y = None
    page.crop_w = None
    page.crop_h = None
    await db.flush()
    return page


@router.post("/pages/{page_id}/crop/auto", response_model=PageOut)
async def auto_crop_page(
    page_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Page:
    """Automatically detect content region and set as crop."""
    from app.ocr import detect_content_bounds, preprocess_image

    stmt = (
        select(Page)
        .join(Document, Page.document_id == Document.id)
        .options(selectinload(Page.ocr_results))
        .where(Page.id == page_id, Document.user_id == current_user.id)
    )
    result = await db.execute(stmt)
    page = result.scalar_one_or_none()
    if page is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Page not found")

    image = preprocess_image(page.image_path, rotation=page.rotation)
    bounds = detect_content_bounds(image)

    if bounds is None:
        # Content covers the whole image — clear any existing crop.
        page.crop_x = None
        page.crop_y = None
        page.crop_w = None
        page.crop_h = None
    else:
        page.crop_x, page.crop_y, page.crop_w, page.crop_h = bounds

    await db.flush()
    return page
