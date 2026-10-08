"""Ink-layer understanding for handwritten pages.

This module adds a safe, modular classification layer above the existing OCR
pipeline. It does not replace OCR; instead it identifies likely regions of
main ink, crossed-out writing, margin notes, and annotation-like noise. The
result is a lightweight structured signal that can be used by later evidence
and risk layers without breaking the baseline workflow.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from PIL import Image

InkLayerLabel = Literal[
    "MAIN_INK",
    "CROSSED_OUT_INK",
    "MARGIN_NOTE",
    "ANNOTATION",
    "NOISE",
    "DIAGRAM_SKETCH",
    "UNKNOWN",
]


@dataclass
class InkRegion:
    """A simple classification for a region of handwriting or page annotation."""

    label: InkLayerLabel
    bbox: tuple[int, int, int, int]
    score: float
    relationship: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "label": self.label,
            "bbox": list(self.bbox),
            "score": self.score,
            "relationship": self.relationship,
            "metadata": self.metadata,
        }
        return payload


class InkLayerAnalyzer:
    """Heuristic ink-layer analyzer.

    The default implementation is intentionally lightweight and robust: it never
    hard-fails OCR if vision heuristics are weak. Instead it produces a safe
    fallback main-ink region and optional additional labels when evidence is
    strong enough.
    """

    def analyze_image(self, image: Image.Image | str) -> list[InkRegion]:
        img = image if isinstance(image, Image.Image) else Image.open(image)
        width, height = img.size
        if width <= 0 or height <= 0:
            return [InkRegion("UNKNOWN", (0, 0, 0, 0), 0.0)]

        regions: list[InkRegion] = [
            InkRegion("MAIN_INK", (0, 0, width, height), 0.75, metadata={"source": "fallback"})
        ]

        margin_region = self._detect_margin_note(img)
        if margin_region is not None:
            regions.append(margin_region)

        revision_region = self._detect_crossed_out_revision(img)
        if revision_region is not None:
            regions.append(revision_region)

        if not any(r.label == "MAIN_INK" for r in regions):
            regions.insert(0, InkRegion("MAIN_INK", (0, 0, width, height), 0.7, metadata={"source": "fallback"}))

        return regions

    def classify_bbox(self, image: Image.Image | str, bbox: tuple[int, int, int, int]) -> InkRegion:
        regions = self.analyze_image(image)
        x, y, w, h = bbox
        for region in regions:
            rx, ry, rw, rh = region.bbox
            overlap = self._overlap_score((x, y, w, h), (rx, ry, rw, rh))
            if overlap > 0.25:
                return region
        return InkRegion("MAIN_INK", bbox, 0.7, metadata={"source": "fallback"})

    def _detect_margin_note(self, image: Image.Image) -> InkRegion | None:
        gray = image.convert("L")
        arr = np.asarray(gray)
        if arr.size == 0:
            return None

        width, height = image.size
        margin_width = max(10, width // 8)
        left = arr[:, :margin_width]
        right = arr[:, -margin_width:]
        left_density = float(np.mean(left < 200))
        right_density = float(np.mean(right < 200))

        if max(left_density, right_density) > 0.12:
            bbox = (0, 0, margin_width, height)
            label = "MARGIN_NOTE"
            if left_density > right_density:
                bbox = (0, 0, margin_width, height)
            else:
                bbox = (width - margin_width, 0, margin_width, height)
            return InkRegion(
                label=label,
                bbox=bbox,
                score=min(0.95, max(left_density, right_density) * 4.0),
                metadata={"heuristic": "margin_density"},
            )
        return None

    def _detect_crossed_out_revision(self, image: Image.Image) -> InkRegion | None:
        gray = image.convert("L")
        arr = np.asarray(gray)
        if arr.size == 0:
            return None

        width, height = image.size
        dark = arr < 200
        row_counts = dark.sum(axis=1)
        candidate_rows = [i for i, count in enumerate(row_counts) if count > width * 0.35]
        if len(candidate_rows) < 3:
            return None

        y_start = candidate_rows[0]
        y_end = candidate_rows[-1]
        bbox = (0, max(y_start - 3, 0), width, max(8, y_end - y_start + 6))

        return InkRegion(
            label="CROSSED_OUT_INK",
            bbox=bbox,
            score=0.82,
            relationship={
                "type": "revision",
                "deleted_text": "unknown",
                "replacement_text": "unknown",
                "source_regions": [list(bbox)],
            },
            metadata={"heuristic": "horizontal_stroke"},
        )

    @staticmethod
    def _overlap_score(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        intersection = (x2 - x1) * (y2 - y1)
        union = max(1, aw * ah + bw * bh - intersection)
        return intersection / union


def classify_ink_regions(image: Image.Image | str) -> list[InkRegion]:
    """Public helper for the rest of the app.

    This is intentionally conservative: the system is allowed to return a
    fallback main-ink region even when it cannot confidently separate all ink
    types. That keeps the pipeline reliable while providing a useful structure
    for later research layers.
    """
    return InkLayerAnalyzer().analyze_image(image)


def serialize_ink_regions(regions: list[InkRegion]) -> str:
    return json.dumps([region.to_dict() for region in regions], ensure_ascii=False)
