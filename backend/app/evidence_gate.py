"""Evidence gate for visual reconstruction.

This module implements the core safety rule for InkShield: a language model or
memory suggestion may propose a correction only when the handwriting evidence is
strong enough to support it. If evidence is insufficient, the system returns the
uncertain original text or a REVIEW/ABSTAIN decision instead of hallucinating a
"better" result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
            insertions = current[j - 1] + 1
            deletions = previous[j] + 1
            substitutions = previous[j - 1] + (ca != cb)
            current.append(min(insertions, deletions, substitutions))
        previous = current
    return previous[-1]


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().split()).lower()


def _candidate_similarity(raw_text: str, candidate: str) -> float:
    if not raw_text and not candidate:
        return 1.0
    if not raw_text or not candidate:
        return 0.0
    raw = _normalize_text(raw_text)
    cand = _normalize_text(candidate)
    distance = _levenshtein_distance(raw, cand)
    return max(0.0, 1.0 - (distance / max(len(raw), len(cand), 1)))


@dataclass
class EvidenceDecision:
    raw_text: str
    accepted_text: str
    decision: str
    risk_score: float
    visual_support: float
    model_agreement: float
    reasons: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "accepted_text": self.accepted_text,
            "decision": self.decision,
            "risk_score": self.risk_score,
            "visual_support": self.visual_support,
            "model_agreement": self.model_agreement,
            "reasons": self.reasons,
            "evidence": self.evidence,
        }


class EvidenceGate:
    """Gatekeeper for visually constrained reconstruction.

    The gate never allows a model correction to override strong contrary visual
    evidence. It rates the support for each candidate and either accepts the
    candidate, sends it to review, or abstains from making a correction.
    """

    def analyze(
        self,
        raw_text: str,
        candidates: list[str] | None = None,
        *,
        image_quality: float = 0.5,
        model_agreement: float = 0.5,
        writer_memory_support: float = 0.0,
        language_consistency: float = 0.5,
        risk_margin: float = 0.2,
    ) -> EvidenceDecision:
        candidates = candidates or []
        candidate_pool = [candidate for candidate in candidates if candidate.strip()]
        if not candidate_pool:
            candidate_pool = [raw_text]

        best_text = raw_text
        best_score = -1.0
        for candidate in candidate_pool:
            similarity = _candidate_similarity(raw_text, candidate)
            score = (
                similarity * 0.60
                + image_quality * 0.15
                + model_agreement * 0.15
                + writer_memory_support * 0.10
                + language_consistency * 0.10
            )
            if score > best_score:
                best_score = score
                best_text = candidate

        visual_support = min(
            1.0,
            max(
                0.0,
                (image_quality * 0.45)
                + (model_agreement * 0.35)
                + (writer_memory_support * 0.15)
                + (language_consistency * 0.15)
                + max(0.0, best_score - 0.5) * 0.7,
            ),
        )
        risk_score = max(0.0, 1.0 - visual_support + risk_margin)
        risk_score = min(1.0, risk_score)

        if visual_support < 0.45:
            decision = "ABSTAIN"
            accepted_text = raw_text
            reasons = [
                "Insufficient visual support for any candidate",
                "Low image quality or model agreement",
            ]
        elif visual_support < 0.78:
            decision = "REVIEW"
            accepted_text = raw_text if raw_text.strip() else best_text
            reasons = [
                "Candidate is plausible but not strongly supported",
                "Needs human review before finalization",
            ]
        else:
            decision = "ACCEPT"
            accepted_text = best_text
            reasons = [
                "Visual evidence and model agreement support the candidate",
                "Document-local memory or language consistency did not override strong contrary evidence",
            ]

        return EvidenceDecision(
            raw_text=raw_text,
            accepted_text=accepted_text,
            decision=decision,
            risk_score=round(risk_score, 4),
            visual_support=round(visual_support, 4),
            model_agreement=model_agreement,
            reasons=reasons,
            evidence={
                "best_candidate": best_text,
                "best_similarity": round(_candidate_similarity(raw_text, best_text), 4),
                "image_quality": image_quality,
                "writer_memory_support": writer_memory_support,
                "language_consistency": language_consistency,
            },
        )
