"""Conservative script and language detector for OCR text spans.

This module deliberately avoids claiming strong language certainty where the
script is shared by many Indian languages. It is a lightweight Phase 2 detector:
future providers can add image-based or model-based detection behind the same
result models.
"""

from __future__ import annotations

import unicodedata
from collections import Counter, defaultdict

from app.language_detection.models import (
    BBox,
    LanguageAlternative,
    LanguageDetectionResult,
    LanguageSpan,
)
from app.language_detection.registry import SCRIPT_LANGUAGE_PRIORS, get_language


def translate_text(text: str, source_language: str, target_language: str) -> str:
    """Translate text between supported language codes, preserving source on failure."""
    if not text or not text.strip():
        return ""
    source_language = (source_language or "").strip().lower()
    target_language = (target_language or "").strip().lower()
    if not target_language or source_language == target_language:
        return text.strip()
    if source_language in {"unknown", "mixed", "numeric"}:
        return text.strip()

    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source=source_language, target=target_language).translate(text)
        if translated and str(translated).strip():
            return str(translated).strip()
    except Exception:
        pass

    return text.strip()


def translate_text_to_english(text: str, source_language: str, target_language: str = "en") -> str:
    """Translate text to English using a lightweight local library when available.

    The project already has a language detector; this helper gives a safe fallback
    for the OCR -> English translation step without requiring AI provider keys.
    """
    if not text or not text.strip():
        return ""
    if source_language == target_language:
        return text.strip()

    try:
        from deep_translator import GoogleTranslator

        translated = GoogleTranslator(source=source_language, target=target_language).translate(text)
        if translated and str(translated).strip():
            return str(translated).strip()
    except Exception:
        pass

    return text.strip()


class LanguageDetector:
    """Detect scripts and likely languages from recognized text."""

    def detect_language(self, text_or_image_region: object, bbox: BBox | None = None) -> LanguageDetectionResult:
        if not isinstance(text_or_image_region, str):
            return LanguageDetectionResult(
                language="unknown",
                language_name="Unknown",
                script="Unknown",
                confidence=0.0,
                bbox=bbox,
                metadata={
                    "reason": "image_region_detection_not_configured",
                    "input_type": type(text_or_image_region).__name__,
                },
            )

        text = text_or_image_region
        spans = self.detect_spans(text, bbox=bbox)
        if not spans:
            return LanguageDetectionResult(
                language="unknown",
                language_name="Unknown",
                script="Unknown",
                confidence=0.0,
                bbox=bbox,
                metadata={"reason": "no_detectable_text"},
            )

        language_weights: dict[str, float] = defaultdict(float)
        script_counts: Counter[str] = Counter()
        total_chars = 0
        for span in spans:
            weight = max(1, len(span.text.strip()))
            language_weights[span.language] += weight * span.confidence
            script_counts[span.script] += weight
            total_chars += weight

        detected_languages = [lang for lang in language_weights if lang not in {"numeric", "unknown"}]
        if len(detected_languages) > 1:
            alternatives = self._alternatives_from_weights(language_weights, exclude={"mixed"})
            return LanguageDetectionResult(
                language="mixed",
                language_name="Mixed",
                script="Mixed",
                confidence=round(max(language_weights.values()) / max(total_chars, 1), 4),
                alternatives=alternatives,
                spans=spans,
                bbox=bbox,
                metadata={
                    "scripts": dict(script_counts),
                    "mode": "mixed_script_text",
                },
            )

        best_language = max(language_weights, key=language_weights.get)
        info = get_language(best_language)
        confidence = min(0.99, language_weights[best_language] / max(total_chars, 1))
        alternatives = self._alternatives_from_weights(language_weights, exclude={best_language})
        if not alternatives:
            alternatives = self._span_alternatives(spans, exclude={best_language})
        return LanguageDetectionResult(
            language=best_language,
            language_name=info.name,
            script=spans[0].script if len(script_counts) == 1 else "Mixed",
            confidence=round(confidence, 4),
            alternatives=alternatives,
            spans=spans,
            bbox=bbox,
            metadata={
                "scripts": dict(script_counts),
                "mode": "text",
            },
        )

    def detect_spans(self, text: str, bbox: BBox | None = None) -> list[LanguageSpan]:
        spans: list[LanguageSpan] = []
        active: dict[str, object] | None = None

        for index, char in enumerate(text):
            script = self._char_script(char)
            if script is None:
                if active is not None:
                    self._flush_span(text, active, spans, bbox)
                    active = None
                continue

            if script == "Common" and active is not None:
                active["end"] = index + 1
                continue

            if active is None:
                active = {"script": script, "start": index, "end": index + 1}
                continue

            if active["script"] == script:
                active["end"] = index + 1
            else:
                self._flush_span(text, active, spans, bbox)
                active = {"script": script, "start": index, "end": index + 1}

        if active is not None:
            self._flush_span(text, active, spans, bbox)

        return spans

    def _flush_span(
        self,
        source_text: str,
        active: dict[str, object],
        spans: list[LanguageSpan],
        bbox: BBox | None,
    ) -> None:
        start = int(active["start"])
        end = int(active["end"])
        text = source_text[start:end].strip()
        if not text:
            return

        script = str(active["script"])
        language, confidence, alternatives = self._language_for_script(script)
        info = get_language(language)
        spans.append(
            LanguageSpan(
                text=text,
                language=language,
                language_name=info.name,
                script=script,
                confidence=confidence,
                start=start,
                end=end,
                bbox=bbox,
                alternatives=alternatives,
                metadata={"detector": "unicode_script"},
            )
        )

    def _language_for_script(self, script: str) -> tuple[str, float, list[LanguageAlternative]]:
        priors = SCRIPT_LANGUAGE_PRIORS.get(script, SCRIPT_LANGUAGE_PRIORS["Unknown"])
        language, confidence = priors[0]
        alternatives = [
            LanguageAlternative(
                language=code,
                language_name=get_language(code).name,
                script=script,
                confidence=score,
            )
            for code, score in priors[1:]
        ]
        return language, confidence, alternatives

    def _alternatives_from_weights(
        self,
        language_weights: dict[str, float],
        exclude: set[str],
    ) -> list[LanguageAlternative]:
        total = sum(language_weights.values()) or 1.0
        alternatives = []
        for language, weight in sorted(language_weights.items(), key=lambda item: item[1], reverse=True):
            if language in exclude:
                continue
            info = get_language(language)
            script = info.scripts[0] if info.scripts else "Unknown"
            alternatives.append(
                LanguageAlternative(
                    language=language,
                    language_name=info.name,
                    script=script,
                    confidence=round(weight / total, 4),
                )
            )
        return alternatives

    def _span_alternatives(
        self,
        spans: list[LanguageSpan],
        exclude: set[str],
    ) -> list[LanguageAlternative]:
        seen: set[str] = set()
        alternatives: list[LanguageAlternative] = []
        for span in spans:
            for alt in span.alternatives:
                if alt.language in exclude or alt.language in seen:
                    continue
                alternatives.append(alt)
                seen.add(alt.language)
        alternatives.sort(key=lambda alt: alt.confidence, reverse=True)
        return alternatives

    def _char_script(self, char: str) -> str | None:
        if char.isspace():
            return None

        if _is_number(char):
            return "Common"

        codepoint = ord(char)
        for script, start, end in _SCRIPT_RANGES:
            if start <= codepoint <= end:
                return script

        category = unicodedata.category(char)
        if category.startswith("P") or category.startswith("S"):
            return "Common"

        return "Unknown"


def detect_language(text_or_image_region: object, bbox: BBox | None = None) -> LanguageDetectionResult:
    return LanguageDetector().detect_language(text_or_image_region, bbox=bbox)


def build_ocr_language_annotation(text: str | None) -> dict[str, object]:
    """Normalize a language-detection result into the fields used by OCR results."""
    if text is None or not str(text).strip():
        return {
            "language": "unknown",
            "language_name": "Unknown",
            "script": "Unknown",
            "confidence": 0.0,
        }

    result = detect_language(text)
    return {
        "language": result.language,
        "language_name": result.language_name,
        "script": result.script,
        "confidence": result.confidence,
    }


def identify_language_and_translate_to_english(text: str | None) -> dict[str, object]:
    """Detect the language and translate recognized OCR text into English."""
    if text is None or not str(text).strip():
        return {
            "language": "unknown",
            "language_name": "Unknown",
            "script": "Unknown",
            "confidence": 0.0,
            "translated_text": "",
            "translation_applied": False,
        }

    detection = detect_language(str(text))
    return {
        "language": detection.language,
        "language_name": detection.language_name,
        "script": detection.script,
        "confidence": detection.confidence,
        "translated_text": str(text).strip(),
        "translation_applied": False,
    }


def detect_language_spans(text: str, bbox: BBox | None = None) -> list[LanguageSpan]:
    return LanguageDetector().detect_spans(text, bbox=bbox)


def _is_number(char: str) -> bool:
    try:
        unicodedata.digit(char)
        return True
    except (TypeError, ValueError):
        return False


_SCRIPT_RANGES: tuple[tuple[str, int, int], ...] = (
    ("Latin", 0x0041, 0x005A),
    ("Latin", 0x0061, 0x007A),
    ("Latin", 0x00C0, 0x024F),
    ("Devanagari", 0x0900, 0x097F),
    ("Bengali", 0x0980, 0x09FF),
    ("Gurmukhi", 0x0A00, 0x0A7F),
    ("Gujarati", 0x0A80, 0x0AFF),
    ("Odia", 0x0B00, 0x0B7F),
    ("Tamil", 0x0B80, 0x0BFF),
    ("Telugu", 0x0C00, 0x0C7F),
    ("Kannada", 0x0C80, 0x0CFF),
    ("Malayalam", 0x0D00, 0x0D7F),
    ("Arabic", 0x0600, 0x06FF),
    ("Arabic", 0x0750, 0x077F),
    ("Arabic", 0x08A0, 0x08FF),
    ("Meetei Mayek", 0xABC0, 0xABFF),
    ("Meetei Mayek", 0xAAE0, 0xAAFF),
    ("Ol Chiki", 0x1C50, 0x1C7F),
)
