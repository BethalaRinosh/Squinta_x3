"""Structured language-detection result models.

These dataclasses are intentionally framework-agnostic so they can be reused by
OCR routing, translation, tests, and later API schemas without creating import
cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class LanguageAlternative:
    language: str
    language_name: str
    script: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "language_name": self.language_name,
            "script": self.script,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class LanguageSpan:
    text: str
    language: str
    language_name: str
    script: str
    confidence: float
    start: int
    end: int
    bbox: BBox | None = None
    alternatives: list[LanguageAlternative] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "language": self.language,
            "language_name": self.language_name,
            "script": self.script,
            "confidence": self.confidence,
            "start": self.start,
            "end": self.end,
            "bbox": list(self.bbox) if self.bbox else None,
            "alternatives": [alt.to_dict() for alt in self.alternatives],
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class LanguageDetectionResult:
    language: str
    language_name: str
    script: str
    confidence: float
    alternatives: list[LanguageAlternative] = field(default_factory=list)
    spans: list[LanguageSpan] = field(default_factory=list)
    bbox: BBox | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "language_name": self.language_name,
            "script": self.script,
            "confidence": self.confidence,
            "alternatives": [alt.to_dict() for alt in self.alternatives],
            "spans": [span.to_dict() for span in self.spans],
            "bbox": list(self.bbox) if self.bbox else None,
            "metadata": self.metadata,
        }
