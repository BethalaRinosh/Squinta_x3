import os
from pathlib import Path

from app.config import Settings
from app.routes.ocr import _can_start_ocr, _normalize_image_path
from app.schemas import OcrResultOut


def test_can_start_ocr_blocks_duplicate_processing():
    assert _can_start_ocr(None) is True
    assert _can_start_ocr("idle") is True
    assert _can_start_ocr("processing") is False
    assert _can_start_ocr("done") is False


def test_normalize_image_path_handles_mixed_separators():
    raw = ".\\data\\uploads\\1\\abc123.jpeg"
    normalized = _normalize_image_path(raw)
    expected = os.path.normpath("./data/uploads/1/abc123.jpeg")
    assert normalized == expected


def test_settings_make_relative_storage_paths_absolute():
    settings = Settings(UPLOAD_DIR="./data/uploads", MODEL_DIR="./data/models")
    project_root = Path(__file__).resolve().parents[2]

    assert Path(settings.UPLOAD_DIR).is_absolute()
    assert Path(settings.UPLOAD_DIR) == (project_root / "data" / "uploads").resolve()
    assert Path(settings.MODEL_DIR).is_absolute()
    assert Path(settings.MODEL_DIR) == (project_root / "data" / "models").resolve()


def test_ocr_result_schema_exposes_translated_english_text():
    result = OcrResultOut(
        id=1,
        page_id=2,
        bbox_x=0,
        bbox_y=0,
        bbox_w=10,
        bbox_h=10,
        text="நாளைக்கு சந்திப்பு",
        confidence=0.95,
        language="ta",
        language_name="Tamil",
        script="Tamil",
        language_confidence=0.9,
        translated_text="Meet tomorrow",
        translation_applied=True,
        created_at="2026-01-01T00:00:00Z",
    )

    assert result.translated_text == "Meet tomorrow"
    assert result.translation_applied is True
