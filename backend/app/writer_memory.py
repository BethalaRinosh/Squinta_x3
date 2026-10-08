"""Document-local writer memory.

This module stores repeated names, technical terms, and recurring tokens already
seen within a document. It can suggest candidates for uncertain OCR text, but it
never overrides visual evidence or drives a confident correction on its own.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


def _normalize_token(value: str) -> str:
    return " ".join(value.strip().split()).lower()


def _levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (ca != cb),
                )
            )
        previous = current
    return previous[-1]


def _similarity(a: str, b: str) -> float:
    a = _normalize_token(a)
    b = _normalize_token(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    distance = _levenshtein_distance(a, b)
    return max(0.0, 1.0 - (distance / max(len(a), len(b), 1)))


@dataclass
class WriterMemoryResult:
    term: str
    similarity: float
    source: str = "document_memory"


class DocumentLocalWriterMemory:
    """Per-document vocabulary and term memory.

    This is intentionally constrained: it only suggests candidates from words
    already seen in the same document. A candidate never bypasses the visual
    evidence gate.
    """

    def __init__(self) -> None:
        self._documents: dict[int, list[str]] = defaultdict(list)
        self._term_counts: dict[str, int] = defaultdict(int)

    def register_document(self, document_id: int, texts: list[str]) -> None:
        for text in texts:
            for token in self._extract_terms(text):
                self._documents[document_id].append(token)
                self._term_counts[token] += 1

    def _extract_terms(self, text: str) -> list[str]:
        tokens = []
        for chunk in text.replace("/", " ").replace("-", " ").split():
            clean = chunk.strip(".,;:!?()[]{}\"'")
            if len(clean) >= 2:
                tokens.append(clean)
        return tokens

    def suggest_candidates(self, term: str, document_id: int | None = None, limit: int = 5) -> list[str]:
        target = _normalize_token(term)
        if not target:
            return []

        candidates: list[tuple[str, float]] = []
        for stored_term, count in self._term_counts.items():
            if stored_term == target:
                continue
            similarity = _similarity(target, stored_term)
            if similarity >= 0.62:
                candidates.append((stored_term, similarity * (1.0 + min(count, 10) / 20.0)))

        if document_id is not None:
            doc_terms = set(self._documents.get(document_id, []))
            candidates = [
                (term, score)
                for term, score in candidates
                if term in doc_terms
            ]

        candidates.sort(key=lambda item: item[1], reverse=True)
        return [term for term, _ in candidates[:limit]]

    def evaluate_candidate(
        self,
        raw_term: str,
        candidate: str,
        *,
        visual_support: float = 0.5,
        risk_margin: float = 0.2,
        document_id: int | None = None,
    ) -> dict[str, Any]:
        suggestions = self.suggest_candidates(raw_term, document_id=document_id)
        allowed = bool(
            candidate in suggestions
            and visual_support >= 0.5
            and risk_margin <= 0.35
        )
        return {
            "allowed": allowed,
            "source": "document_memory",
            "suggestions": suggestions,
            "visual_guard": bool(visual_support >= 0.5),
            "risk_guard": bool(risk_margin <= 0.35),
        }
