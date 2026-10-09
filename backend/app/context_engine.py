"""
Domain-aware OCR context engine.

The engine detects likely document domains from OCR candidates and provides
domain vocabulary/context back to the vision model. It deliberately does NOT
overwrite OCR text on its own. Gemini remains the recognition engine; this
module supplies constrained domain evidence so ambiguous handwriting can be
resolved using terminology that is plausible for the detected domain.

Medical terminology is backed by a compact local seed lexicon rather than a
large external download. The design is intentionally provider-agnostic so a
larger ontology such as UMLS/MeSH/RxNorm can be plugged in later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class DomainProfile:
    name: str
    aliases: tuple[str, ...]
    vocabulary: tuple[str, ...]
    context: str


DOMAIN_PROFILES: tuple[DomainProfile, ...] = (
    DomainProfile(
        name="medical",
        aliases=("medical", "medicine", "clinical", "doctor", "hospital", "prescription", "patient"),
        vocabulary=(
            "diagnosis", "symptom", "patient", "history", "examination", "assessment",
            "treatment", "prescription", "medication", "dose", "dosage", "tablet",
            "capsule", "injection", "blood", "pressure", "pulse", "temperature",
            "heart rate", "respiratory", "oxygen", "saturation", "fever", "cough",
            "pain", "nausea", "vomiting", "diabetes", "hypertension", "hypotension",
            "asthma", "bronchitis", "pneumonia", "infection", "antibiotic",
            "amoxicillin", "azithromycin", "metformin", "insulin", "paracetamol",
            "acetaminophen", "ibuprofen", "mg", "ml", "mcg", "bpm", "mmhg",
            "ecg", "ekg", "cbc", "wbc", "rbc", "hemoglobin", "creatinine",
            "glucose", "cholesterol", "follow-up", "discharge", "allergy",
        ),
        context=(
            "This appears to be medical/clinical handwriting. Preserve drug names, "
            "dosages, units, abbreviations, anatomy, diagnoses, symptoms and clinical "
            "measurements exactly when they are visually supported. Prefer established "
            "medical terms over ordinary words when the handwriting is ambiguous, but "
            "never invent a term that is not supported by the image."
        ),
    ),
    DomainProfile(
        name="legal",
        aliases=("legal", "law", "court", "contract", "agreement", "attorney", "lawyer"),
        vocabulary=(
            "plaintiff", "defendant", "petitioner", "respondent", "affidavit",
            "jurisdiction", "contract", "agreement", "clause", "section", "subsection",
            "whereas", "hereby", "thereof", "indemnity", "liability", "breach",
            "notice", "court", "appeal", "order", "witness", "testimony",
        ),
        context=(
            "This appears to be legal writing. Preserve section numbers, clause labels, "
            "case terminology, names, dates and legal abbreviations. Prefer legal terms "
            "when handwriting is ambiguous, but only when the visual evidence supports them."
        ),
    ),
    DomainProfile(
        name="finance",
        aliases=("finance", "financial", "bank", "accounting", "investment", "trading", "tax"),
        vocabulary=(
            "invoice", "revenue", "expense", "profit", "loss", "asset", "liability",
            "equity", "debit", "credit", "balance", "interest", "principal", "loan",
            "account", "tax", "gst", "vat", "dividend", "yield", "portfolio",
            "stock", "bond", "market", "cash flow", "ebitda", "roi", "p&l",
        ),
        context=(
            "This appears to be financial/accounting writing. Preserve amounts, "
            "currencies, percentages, account identifiers, dates and financial "
            "abbreviations. Prefer accounting/finance terminology when ambiguity is "
            "caused by handwriting, without fabricating values."
        ),
    ),
    DomainProfile(
        name="science",
        aliases=("science", "scientific", "research", "laboratory", "lab", "biology", "chemistry", "physics"),
        vocabulary=(
            "hypothesis", "experiment", "sample", "control", "variable", "method",
            "observation", "result", "analysis", "equation", "reaction", "molecule",
            "enzyme", "protein", "cell", "species", "velocity", "force", "mass",
            "energy", "temperature", "pressure", "concentration", "solution",
        ),
        context=(
            "This appears to be scientific/technical writing. Preserve symbols, "
            "units, equations, scientific names and technical abbreviations exactly. "
            "Use scientific vocabulary to resolve ambiguous handwriting only when "
            "supported by visible context."
        ),
    ),
    DomainProfile(
        name="education",
        aliases=("education", "school", "college", "university", "lecture", "class", "exam", "notes"),
        vocabulary=(
            "assignment", "lecture", "chapter", "topic", "question", "answer", "example",
            "definition", "theorem", "proof", "formula", "algorithm", "semester",
            "exam", "marks", "grade", "project", "notes",
        ),
        context=(
            "This appears to be educational/class notes. Preserve subject terminology, "
            "formulas, numbered questions and examples. Use academic vocabulary to help "
            "resolve ambiguous words when supported by surrounding text."
        ),
    ),
)


def _normalise(text: str) -> str:
    """Normalize text while keeping clinically useful symbols and word boundaries."""
    return re.sub(r"[^a-z0-9%+./-]+", " ", (text or "").lower()).strip()


def _tokenise(text: str) -> set[str]:
    """Return meaningful tokens; short units are retained when medically useful."""
    return {token for token in _normalise(text).split() if len(token) > 2}


def _contains_term(text: str, term: str) -> bool:
    """Match a whole term or phrase, not an arbitrary substring."""
    normalized_text = _normalise(text)
    normalized_term = _normalise(term)
    if not normalized_text or not normalized_term:
        return False
    return bool(re.search(
        rf"(?<![a-z0-9]){re.escape(normalized_term)}(?![a-z0-9])",
        normalized_text,
    ))


def _profile_evidence(profile: DomainProfile, text: str) -> dict:
    """Collect distinct weighted signals for a domain without double-counting fragments."""
    alias_hits = [alias for alias in profile.aliases if _contains_term(text, alias)]
    vocabulary_hits = [
        term for term in profile.vocabulary
        if _contains_term(text, term)
    ]

    # A domain label or explicit document type is strong evidence. A single
    # generic vocabulary word is weak evidence; multi-word terms are more specific.
    alias_score = sum(3.0 if len(_normalise(alias).split()) > 1 else 2.4 for alias in alias_hits)
    vocabulary_score = sum(
        1.8 if len(_normalise(term).split()) > 1 else 1.0
        for term in vocabulary_hits
    )
    total = alias_score + vocabulary_score
    return {
        "score": total,
        "alias_hits": alias_hits,
        "vocabulary_hits": vocabulary_hits,
        "signal_count": len(alias_hits) + len(vocabulary_hits),
    }


def _profile_score(profile: DomainProfile, text: str) -> float:
    """Compatibility helper returning a length-normalized evidence score."""
    evidence = _profile_evidence(profile, text)
    token_count = max(1, len(_normalise(text).split()))
    return evidence["score"] / max(1.0, min(10.0, token_count * 0.22))


def _confidence_from_evidence(best: dict, runner_up_score: float, token_count: int) -> float:
    """Conservative confidence based on amount and distinctiveness of evidence."""
    score = float(best["score"])
    signals = int(best["signal_count"])
    if score <= 0 or signals == 0:
        return 0.0

    # Require more than one weak clue. Strong explicit aliases can still
    # identify short forms such as "medical prescription".
    evidence_strength = min(1.0, score / 5.0)
    evidence_breadth = min(1.0, signals / 3.0)
    separation = max(0.0, min(1.0, (score - runner_up_score) / max(score, 1.0)))
    confidence = 0.55 * evidence_strength + 0.25 * evidence_breadth + 0.20 * separation

    if signals == 1 and score < 2.4:
        confidence = min(confidence, 0.35)
    if signals == 1 and token_count <= 3:
        confidence = min(confidence, 0.60)
    return round(max(0.0, min(1.0, confidence)), 4)


def detect_domain(text: str, minimum_score: float = 0.62) -> tuple[str, float]:
    """Return (domain, confidence), using conservative, phrase-aware evidence."""
    if not text or not text.strip():
        return "general", 0.0

    evidence = [
        (profile, _profile_evidence(profile, text))
        for profile in DOMAIN_PROFILES
    ]
    evidence.sort(key=lambda item: item[1]["score"], reverse=True)
    if not evidence or evidence[0][1]["score"] <= 0:
        return "general", 0.0

    best_profile, best = evidence[0]
    runner_up_score = evidence[1][1]["score"] if len(evidence) > 1 else 0.0
    confidence = _confidence_from_evidence(
        best, runner_up_score, len(_normalise(text).split())
    )
    if confidence < minimum_score:
        return "general", 0.0
    return best_profile.name, confidence

def get_profile(domain: str) -> DomainProfile | None:
    domain = (domain or "").strip().lower()
    for profile in DOMAIN_PROFILES:
        if profile.name == domain:
            return profile
    return None


def select_vocabulary(text: str, domain: str, limit: int = 24) -> list[str]:
    """Return domain terms supported by whole-word/phrase clues in the candidate."""
    profile = get_profile(domain)
    if profile is None or not text:
        return []

    scored: list[tuple[int, str]] = []
    for term in profile.vocabulary:
        if _contains_term(text, term):
            specificity = 2 if len(_normalise(term).split()) > 1 else 1
            scored.append((specificity, term))

    scored.sort(key=lambda item: (-item[0], len(item[1])))
    return [term for _, term in scored[:max(0, limit)]]

def build_context(text: str, forced_domain: str | None = None) -> dict:
    """Build a compact context payload with evidence and ambiguity diagnostics."""
    normalized_forced = (forced_domain or "").strip().lower()
    evidence = [
        (profile, _profile_evidence(profile, text))
        for profile in DOMAIN_PROFILES
    ]
    evidence.sort(key=lambda item: item[1]["score"], reverse=True)

    if normalized_forced and normalized_forced != "auto":
        profile = get_profile(normalized_forced)
        if profile is None:
            return {
                "domain": "general",
                "confidence": 0.0,
                "context": "General handwriting OCR. Transcribe only what is visually supported.",
                "vocabulary": [],
                "evidence": [],
                "alternatives": [],
                "forced": False,
            }
        domain = profile.name
        confidence = 1.0
        best_evidence = next((item[1] for item in evidence if item[0].name == domain), {})
        forced = True
    else:
        if not evidence or evidence[0][1]["score"] <= 0:
            return {
                "domain": "general",
                "confidence": 0.0,
                "context": "General handwriting OCR. Transcribe only what is visually supported.",
                "vocabulary": [],
                "evidence": [],
                "alternatives": [],
                "forced": False,
            }
        best_profile, best_evidence = evidence[0]
        runner_up_score = evidence[1][1]["score"] if len(evidence) > 1 else 0.0
        confidence = _confidence_from_evidence(
            best_evidence, runner_up_score, len(_normalise(text).split())
        )
        if confidence < 0.62:
            return {
                "domain": "general",
                "confidence": round(confidence, 4),
                "context": "Domain evidence is weak or ambiguous. Transcribe conservatively without assuming a specialist domain.",
                "vocabulary": [],
                "evidence": [],
                "alternatives": [
                    {"domain": profile.name, "score": round(float(item["score"]), 2)}
                    for profile, item in evidence[:2] if item["score"] > 0
                ],
                "forced": False,
            }
        domain = best_profile.name
        forced = False

    profile = get_profile(domain)
    if profile is None:
        return {
            "domain": "general",
            "confidence": 0.0,
            "context": "General handwriting OCR. Transcribe only what is visually supported.",
            "vocabulary": [],
            "evidence": [],
            "alternatives": [],
            "forced": False,
        }

    terms = select_vocabulary(text, domain)
    runner_up_score = next(
        (float(item["score"]) for other_profile, item in evidence if other_profile.name != domain),
        0.0,
    )
    return {
        "domain": domain,
        "confidence": round(confidence, 4),
        "context": profile.context,
        "vocabulary": terms or list(profile.vocabulary[:12]),
        "evidence": list(dict.fromkeys(
            best_evidence.get("alias_hits", []) + best_evidence.get("vocabulary_hits", [])
        ))[:12],
        "alternatives": [
            {"domain": other_profile.name, "score": round(float(item["score"]), 2)}
            for other_profile, item in evidence if other_profile.name != domain and item["score"] > 0
        ][:2],
        "score_margin": round(float(best_evidence.get("score", 0.0)) - runner_up_score, 2),
        "forced": forced,
    }

def build_context_prompt(text: str, context: dict) -> str:
    """Render bounded, evidence-backed guidance for a downstream vision model."""
    domain = context.get("domain", "general")
    confidence = float(context.get("confidence", 0.0))
    terms = context.get("vocabulary") or []
    guidance = context.get("context") or "Use only visually supported text."
    evidence = context.get("evidence") or []
    alternatives = context.get("alternatives") or []
    evidence_text = ", ".join(str(term) for term in evidence[:12]) or "none"
    alternatives_text = ", ".join(
        f"{item.get('domain')} (evidence score {item.get('score')})"
        for item in alternatives[:2]
    ) or "none"

    return (
        f"DOMAIN HYPOTHESIS: {domain} (confidence {confidence:.2f})\n"
        f"DOMAIN GUIDANCE: {guidance}\n"
        f"OBSERVED DOMAIN CLUES IN CANDIDATE: {evidence_text}\n"
        f"RELEVANT VOCABULARY: {', '.join(str(term) for term in terms[:24]) if terms else 'none'}\n"
        f"COMPETING DOMAINS: {alternatives_text}\n"
        "Use this domain only as a spelling/terminology hint. The image is the source of truth. "
        "Do not infer missing words, diagnoses, medicine names, doses, dates, names, or numbers. "
        "Do not normalize or 'correct' a clinically plausible value unless the strokes in the image support it. "
        "If the image cannot distinguish candidates, preserve uncertainty rather than selecting the most common term."
    )
