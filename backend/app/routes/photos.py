"""Google Photos Picker API integration.

The old Library API scopes (photoslibrary.readonly) were removed in March 2025.
This module uses the Photos Picker API instead:
1. Backend creates a picker session
2. Frontend opens the pickerUri for the user to select photos
3. Frontend polls the session until the user finishes
4. Backend fetches the selected media items and imports them
"""

import os
import uuid
from typing import List, Optional

import aiofiles
import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.auth import get_current_user
from app.config import settings
from app.database import get_db
from app.models import Document, Page, User

router = APIRouter(prefix="/photos", tags=["photos"])

PICKER_API = "https://photospicker.googleapis.com/v1"


# ── Schemas ──────────────────────────────────────────────────────────────────


class PickerSessionResponse(BaseModel):
    session_id: str
    picker_uri: str


class PickerPollResponse(BaseModel):
    ready: bool
    media_items_count: int = 0


class PickerImportRequest(BaseModel):
    session_id: str
    document_name: str = "Google Photos Import"


class PickerImportResponse(BaseModel):
    document_id: int
    imported_count: int


# ── Helpers ──────────────────────────────────────────────────────────────────


def _get_google_token(user: User) -> str:
    if not user.google_access_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Google access token not available. Please sign out and sign back in.",
        )
    return user.google_access_token


def _user_upload_dir(user_id: int) -> str:
    path = os.path.join(settings.UPLOAD_DIR, str(user_id))
    os.makedirs(path, exist_ok=True)
    return path


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.post("/picker/session", response_model=PickerSessionResponse)
async def create_picker_session(
    current_user: User = Depends(get_current_user),
) -> PickerSessionResponse:
    """Create a Google Photos Picker session.

    Returns a picker_uri that the frontend should open for the user
    to select photos.
    """
    token = _get_google_token(current_user)

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{PICKER_API}/sessions",
            headers={"Authorization": f"Bearer {token}"},
        )

    if resp.status_code in (401, 403):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Google Photos access denied. Please sign out and sign back in.",
        )
    resp.raise_for_status()

    data = resp.json()
    return PickerSessionResponse(
        session_id=data["id"],
        picker_uri=data["pickerUri"],
    )


@router.get("/picker/session/{session_id}", response_model=PickerPollResponse)
async def poll_picker_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
) -> PickerPollResponse:
    """Poll a picker session to check if the user has finished selecting photos."""
    token = _get_google_token(current_user)

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{PICKER_API}/sessions/{session_id}",
            headers={"Authorization": f"Bearer {token}"},
        )

    if resp.status_code in (401, 403):
        raise HTTPException(status_code=401, detail="Google token expired.")
    resp.raise_for_status()

    data = resp.json()
    media_set = data.get("mediaItemsSet", False)

    return PickerPollResponse(
        ready=media_set,
        media_items_count=0,  # count is not available until we list items
    )


@router.post("/picker/import", response_model=PickerImportResponse)
async def import_from_picker(
    body: PickerImportRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> PickerImportResponse:
    """Import the photos selected in a picker session as a new document."""
    token = _get_google_token(current_user)

    # List all selected media items from the session.
    all_items = []
    page_token: Optional[str] = None

    async with httpx.AsyncClient(timeout=60) as client:
        while True:
            params = {"sessionId": body.session_id, "pageSize": 100}
            if page_token:
                params["pageToken"] = page_token

            resp = await client.get(
                f"{PICKER_API}/mediaItems",
                headers={"Authorization": f"Bearer {token}"},
                params=params,
            )

            if resp.status_code in (401, 403):
                raise HTTPException(status_code=401, detail="Google token expired.")
            resp.raise_for_status()

            data = resp.json()
            for item in data.get("mediaItems", []):
                media_file = item.get("mediaFile", {})
                mime = media_file.get("mimeType", "")
                if mime.startswith("image/"):
                    all_items.append({
                        "id": item.get("id"),
                        "base_url": media_file.get("baseUrl", ""),
                        "filename": media_file.get("filename", "photo.jpg"),
                        "mime_type": mime,
                    })

            page_token = data.get("nextPageToken")
            if not page_token:
                break

    if not all_items:
        raise HTTPException(status_code=404, detail="No images found in picker selection.")

    # Create document.
    document = Document(
        user_id=current_user.id,
        name=body.document_name,
        source="google_photos",
    )
    db.add(document)
    await db.flush()

    upload_dir = _user_upload_dir(current_user.id)
    imported = 0

    async with httpx.AsyncClient(timeout=60) as client:
        for idx, item in enumerate(all_items):
            # baseUrl from Picker API is valid for 60 minutes.
            # Append =d to download the original full-resolution image.
            download_url = item["base_url"]
            if not download_url:
                continue
            if "=" not in download_url.split("/")[-1]:
                download_url += "=d"

            ext = os.path.splitext(item["filename"])[1].lower() or ".jpg"
            unique_name = f"{uuid.uuid4().hex}{ext}"
            dest = os.path.join(upload_dir, unique_name)

            try:
                resp = await client.get(
                    download_url,
                    headers={"Authorization": f"Bearer {token}"},
                )
                resp.raise_for_status()
                async with aiofiles.open(dest, "wb") as f:
                    await f.write(resp.content)
            except httpx.HTTPError:
                continue

            # Convert HEIC/HEIF to JPEG so browsers can display them
            if ext in (".heic", ".heif"):
                from app.routes.documents import _convert_heic_to_jpeg
                dest = _convert_heic_to_jpeg(dest)
            else:
                from app.routes.documents import _bake_exif_rotation
                _bake_exif_rotation(dest)

            page = Page(
                document_id=document.id,
                image_path=dest,
                page_number=idx + 1,
            )
            db.add(page)
            imported += 1

    await db.flush()

    # Auto-trigger OCR on all imported pages.
    from app.routes.ocr import _run_ocr_on_page
    from sqlalchemy.orm import selectinload as _sl
    re_stmt = (
        select(Document)
        .options(_sl(Document.pages))
        .where(Document.id == document.id)
    )
    re_result = await db.execute(re_stmt)
    doc_reloaded = re_result.scalar_one()
    for pg in doc_reloaded.pages:
        pg.processing_status = "processing"
        background_tasks.add_task(_run_ocr_on_page, pg.id, current_user.id)
    await db.flush()

    # Clean up the picker session.
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.delete(
                f"{PICKER_API}/sessions/{body.session_id}",
                headers={"Authorization": f"Bearer {token}"},
            )
    except Exception:
        pass  # best-effort cleanup

    return PickerImportResponse(
        document_id=document.id,
        imported_count=imported,
    )
