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
    return re.sub(r"[^a-z0-9%+./-]+", " ", text.lower()).strip()


def _tokenise(text: str) -> set[str]:
    return {t for t in _normalise(text).split() if len(t) > 2}


def _profile_score(profile: DomainProfile, text: str) -> float:
    normalised = _normalise(text)
    tokens = _tokenise(text)
    if not tokens:
        return 0.0

    alias_hits = sum(1 for alias in profile.aliases if _normalise(alias) in normalised)
    vocab_hits = sum(
        1 for term in profile.vocabulary
        if _normalise(term) in normalised or bool(_tokenise(term) & tokens)
    )

    # Alias signals are strong. Vocabulary gives broad context.
    raw = alias_hits * 3.0 + vocab_hits
    return raw / max(1.0, min(12.0, len(tokens) * 0.35))


def detect_domain(text: str, minimum_score: float = 0.55) -> tuple[str, float]:
    """Return (domain, confidence). Falls back to general."""
    if not text or not text.strip():
        return "general", 0.0

    scored = sorted(
        ((profile.name, _profile_score(profile, text)) for profile in DOMAIN_PROFILES),
        key=lambda item: item[1],
        reverse=True,
    )
    if not scored or scored[0][1] < minimum_score:
        return "general", 0.0

    best_name, best_raw = scored[0]
    confidence = max(0.0, min(1.0, best_raw / 3.0))
    return best_name, confidence


def get_profile(domain: str) -> DomainProfile | None:
    domain = (domain or "").strip().lower()
    for profile in DOMAIN_PROFILES:
        if profile.name == domain:
            return profile
    return None


def select_vocabulary(text: str, domain: str, limit: int = 24) -> list[str]:
    """Return relevant domain terms already hinted at by the OCR candidate."""
    profile = get_profile(domain)
    if profile is None:
        return []

    normalised = _normalise(text)
    tokens = _tokenise(text)
    scored: list[tuple[int, str]] = []

    for term in profile.vocabulary:
        nt = _normalise(term)
        overlap = len(_tokenise(term) & tokens)
        substring = int(nt in normalised)
        if overlap or substring:
            scored.append((overlap * 2 + substring, term))

    scored.sort(key=lambda item: (-item[0], len(item[1])))
    return [term for _, term in scored[:limit]]


def build_context(text: str, forced_domain: str | None = None) -> dict:
    """Build a compact, JSON-friendly context payload."""
    if forced_domain and forced_domain.lower() != "auto":
        domain = forced_domain.lower()
        confidence = 1.0
    else:
        domain, confidence = detect_domain(text)

    profile = get_profile(domain)
    if profile is None:
        return {
            "domain": "general",
            "confidence": round(confidence, 4),
            "context": "General handwriting OCR. Transcribe only what is visually supported.",
            "vocabulary": [],
        }

    terms = select_vocabulary(text, domain)
    return {
        "domain": domain,
        "confidence": round(confidence, 4),
        "context": profile.context,
        "vocabulary": terms or list(profile.vocabulary[:24]),
    }


def build_context_prompt(text: str, context: dict) -> str:
    """Turn context data into safe instructions for a downstream LLM."""
    domain = context.get("domain", "general")
    confidence = float(context.get("confidence", 0.0))
    terms = context.get("vocabulary") or []
    guidance = context.get("context") or "Use only visually supported text."

    return (
        f"DOMAIN CONTEXT: {domain} (confidence {confidence:.2f})\n"
        f"DOMAIN GUIDANCE: {guidance}\n"
        f"RELEVANT VOCABULARY: {', '.join(terms[:24]) if terms else 'none'}\n"
        "The domain context is a hint, not ground truth. Never replace a visually "
        "uncertain word merely because a domain term would be plausible."
    )
