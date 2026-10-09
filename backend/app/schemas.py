import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


# ── User ──────────────────────────────────────────────────────────────────────


class UserOut(BaseModel):
    id: int
    google_id: str
    email: str
    name: str
    picture_url: Optional[str] = None
    created_at: datetime.datetime

    model_config = {"from_attributes": True}


# ── Document ──────────────────────────────────────────────────────────────────


class DocumentCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    source: str = Field(default="upload", pattern=r"^(upload|camera|google_photos)$")


class VisualElementOut(BaseModel):
    id: int
    page_id: int
    element_type: str
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    confidence: float
    label: Optional[str] = None
    geometry: Optional[str] = None
    metadata: Optional[str] = Field(
        default=None,
        validation_alias="element_metadata",
        serialization_alias="metadata",
    )
    created_at: datetime.datetime

    model_config = {"from_attributes": True}


class OcrResultOut(BaseModel):
    id: int
    page_id: int
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    text: str
    confidence: float
    translated_text: Optional[str] = None
    translation_applied: bool = False
    language: Optional[str] = None
    language_name: Optional[str] = None
    script: Optional[str] = None
    language_confidence: Optional[float] = None
    model_version: Optional[str] = None
    ink_layer: str = "MAIN_INK"
    ink_metadata: Optional[str] = None
    struck_through: bool = False
    created_at: datetime.datetime

    model_config = {"from_attributes": True}


class PageOut(BaseModel):
    id: int
    document_id: int
    image_path: str
    image_url: str = ""
    page_number: int
    rotation: int = 0
    crop_x: Optional[int] = None
    crop_y: Optional[int] = None
    crop_w: Optional[int] = None
    crop_h: Optional[int] = None
    page_warped: bool = False
    processing_status: str = "idle"
    created_at: datetime.datetime
    ocr_results: List[OcrResultOut] = []
    visual_elements: List[VisualElementOut] = []

    model_config = {"from_attributes": True}

    def model_post_init(self, __context) -> None:
        """Build a browser URL from the stored upload filesystem path."""
        if self.image_path and not self.image_url:
            from pathlib import Path
            from app.config import settings

            try:
                relative = Path(self.image_path).resolve().relative_to(
                    Path(settings.UPLOAD_DIR).resolve()
                )
                self.image_url = "/api/uploads/" + relative.as_posix()
            except ValueError:
                # Support legacy relative paths already stored in the database.
                path = str(self.image_path).replace("\\", "/")
                if "/uploads/" in path:
                    self.image_url = "/api/uploads/" + path.split("/uploads/", 1)[1].lstrip("/")
                elif path.startswith("data/uploads/"):
                    self.image_url = "/api/uploads/" + path[len("data/uploads/"):]
                else:
                    self.image_url = ""


class DocumentOut(BaseModel):
    id: int
    user_id: int
    name: str
    source: str
    created_at: datetime.datetime
    pages: List[PageOut] = []

    model_config = {"from_attributes": True}


class DocumentListItem(BaseModel):
    """Lightweight document info for list endpoints (no nested pages)."""
    id: int
    user_id: int
    name: str
    source: str
    created_at: datetime.datetime
    page_count: int = 0
    ocr_result_count: int = 0

    model_config = {"from_attributes": True}


# ── Corrections ───────────────────────────────────────────────────────────────


class CorrectionCreate(BaseModel):
    ocr_result_id: int
    corrected_text: str = Field(..., min_length=1)


class CorrectionOut(BaseModel):
    id: int
    ocr_result_id: int
    original_text: str
    corrected_text: str
    corrected_bbox_x: Optional[int] = None
    corrected_bbox_y: Optional[int] = None
    corrected_bbox_w: Optional[int] = None
    corrected_bbox_h: Optional[int] = None
    created_at: datetime.datetime

    model_config = {"from_attributes": True}


# ── Search ────────────────────────────────────────────────────────────────────


class SearchResult(BaseModel):
    ocr_result_id: int
    page_id: int
    page_number: int
    page_image_path: str
    thumbnail_url: str = ""
    document_id: int
    document_name: str
    text: str
    confidence: float
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int

    @property
    def id(self) -> int:
        return self.ocr_result_id

    def model_post_init(self, __context) -> None:
        """Expose a browser URL without leaking the server's filesystem path."""
        if self.thumbnail_url:
            return

        from pathlib import Path
        from app.config import settings

        try:
            relative = Path(self.page_image_path).resolve().relative_to(
                Path(settings.UPLOAD_DIR).resolve()
            )
            self.thumbnail_url = "/api/uploads/" + relative.as_posix()
            return
        except (ValueError, OSError):
            pass

        path = str(self.page_image_path or "").replace("\\\\", "/")
        if "/uploads/" in path:
            path = path.split("/uploads/", 1)[1].lstrip("/")
        elif path.startswith("data/uploads/"):
            path = path[len("data/uploads/"):]
        elif path.startswith("./data/uploads/"):
            path = path[len("./data/uploads/"):]
        else:
            return
        self.thumbnail_url = "/api/uploads/" + path


class SearchResponse(BaseModel):
    query: str
    total: int
    results: List[SearchResult]


# ── Play (correction game) ───────────────────────────────────────────────────


class PlayItem(BaseModel):
    """One item displayed in the correction game UI."""
    ocr_result_id: int
    page_id: int
    page_image_path: str
    image_url: str = ""
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    recognized_text: str
    confidence: float
    document_name: str
    page_rotation: int = 0

    def model_post_init(self, __context) -> None:
        """Convert the filesystem page_image_path to a browser-serveable URL."""
        if self.page_image_path and not self.image_url:
            path = self.page_image_path
            if path.startswith("./data/"):
                path = path[len("./data/"):]
            elif path.startswith("data/"):
                path = path[len("data/"):]
            self.image_url = f"/api/{path}"


class PlayBatchResponse(BaseModel):
    items: List[PlayItem]
    remaining: int  # how many more items are available


class PlaySubmit(BaseModel):
    ocr_result_id: int
    corrected_text: str = Field(..., min_length=1)
    corrected_bbox_x: Optional[int] = None
    corrected_bbox_y: Optional[int] = None
    corrected_bbox_w: Optional[int] = None
    corrected_bbox_h: Optional[int] = None


# ── Model / Fine-tuning ─────────────────────────────────────────────────────


class ModelExportInfo(BaseModel):
    user_id: int
    version: int
    num_corrections_trained: int
    lora_path: Optional[str] = None
    created_at: Optional[datetime.datetime] = None

    model_config = {"from_attributes": True}


class ModelStatusOut(BaseModel):
    has_model: bool
    version: Optional[int] = None
    num_corrections_trained: Optional[int] = None
    total_corrections: int = 0
    corrections_since_last_train: int = 0
    created_at: Optional[datetime.datetime] = None


class TrainRequest(BaseModel):
    """Optional parameters for fine-tuning."""
    epochs: int = Field(default=3, ge=1, le=50)
    learning_rate: float = Field(default=1e-4, gt=0)


class CalibrateRequest(BaseModel):
    """Request body for the calibrate endpoint."""
    page_id: int
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    ground_truth: str = Field(
        default="The quick brown fox jumps over the lazy gray dog.",
        min_length=1,
    )


class TrainResponse(BaseModel):
    message: str
    version: int
    num_corrections: int


# ── Google Photos ─────────────────────────────────────────────────────────────


class GoogleAlbum(BaseModel):
    id: str
    title: str
    media_items_count: Optional[int] = None
    cover_photo_url: Optional[str] = None


class GoogleMediaItem(BaseModel):
    id: str
    filename: str
    mime_type: str
    base_url: str
    width: Optional[int] = None
    height: Optional[int] = None


class PhotoImportRequest(BaseModel):
    album_id: Optional[str] = None
    media_item_ids: Optional[List[str]] = None
    document_name: str = Field(..., min_length=1, max_length=255)


class PhotoImportResponse(BaseModel):
    document: DocumentOut
    imported_count: int


# ── Generic ───────────────────────────────────────────────────────────────────


class MessageResponse(BaseModel):
    message: str
