"""Language identification primitives for InkShield multilingual OCR."""

from app.language_detection.detector import (
    LanguageDetector,
    build_ocr_language_annotation,
    detect_language,
    detect_language_spans,
    identify_language_and_translate_to_english,
    translate_text_to_english,
)
from app.language_detection.models import (
    LanguageAlternative,
    LanguageDetectionResult,
    LanguageSpan,
)
from app.language_detection.registry import SUPPORTED_LANGUAGES, get_language

__all__ = [
    "LanguageAlternative",
    "LanguageDetectionResult",
    "LanguageDetector",
    "LanguageSpan",
    "SUPPORTED_LANGUAGES",
    "build_ocr_language_annotation",
    "detect_language",
    "detect_language_spans",
    "get_language",
    "identify_language_and_translate_to_english",
    "translate_text_to_english",
]
