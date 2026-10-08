"""Language registry for the multilingual InkShield pipeline.

Language support is separated from OCR support and translation support. This
registry only describes target languages, their common scripts, and conservative
script-to-language priors used by the lightweight detector.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LanguageInfo:
    code: str
    name: str
    scripts: tuple[str, ...]
    constitutionally_recognized: bool = True


SUPPORTED_LANGUAGES: dict[str, LanguageInfo] = {
    "as": LanguageInfo("as", "Assamese", ("Bengali",)),
    "bn": LanguageInfo("bn", "Bengali", ("Bengali",)),
    "brx": LanguageInfo("brx", "Bodo", ("Devanagari",)),
    "doi": LanguageInfo("doi", "Dogri", ("Devanagari",)),
    "gu": LanguageInfo("gu", "Gujarati", ("Gujarati",)),
    "hi": LanguageInfo("hi", "Hindi", ("Devanagari",)),
    "kn": LanguageInfo("kn", "Kannada", ("Kannada",)),
    "ks": LanguageInfo("ks", "Kashmiri", ("Arabic", "Devanagari")),
    "kok": LanguageInfo("kok", "Konkani", ("Devanagari",)),
    "mai": LanguageInfo("mai", "Maithili", ("Devanagari",)),
    "ml": LanguageInfo("ml", "Malayalam", ("Malayalam",)),
    "mni": LanguageInfo("mni", "Manipuri", ("Meetei Mayek", "Bengali")),
    "mr": LanguageInfo("mr", "Marathi", ("Devanagari",)),
    "ne": LanguageInfo("ne", "Nepali", ("Devanagari",)),
    "or": LanguageInfo("or", "Odia", ("Odia",)),
    "pa": LanguageInfo("pa", "Punjabi", ("Gurmukhi",)),
    "sa": LanguageInfo("sa", "Sanskrit", ("Devanagari",)),
    "sat": LanguageInfo("sat", "Santali", ("Ol Chiki", "Devanagari")),
    "sd": LanguageInfo("sd", "Sindhi", ("Arabic", "Devanagari")),
    "ta": LanguageInfo("ta", "Tamil", ("Tamil",)),
    "te": LanguageInfo("te", "Telugu", ("Telugu",)),
    "ur": LanguageInfo("ur", "Urdu", ("Arabic",)),
    "en": LanguageInfo("en", "English", ("Latin",), constitutionally_recognized=False),
    "numeric": LanguageInfo("numeric", "Number", ("Common",), constitutionally_recognized=False),
    "unknown": LanguageInfo("unknown", "Unknown", ("Unknown",), constitutionally_recognized=False),
    "mixed": LanguageInfo("mixed", "Mixed", ("Mixed",), constitutionally_recognized=False),
}


SCRIPT_LANGUAGE_PRIORS: dict[str, tuple[tuple[str, float], ...]] = {
    "Latin": (("en", 0.95),),
    "Tamil": (("ta", 0.96),),
    "Malayalam": (("ml", 0.96),),
    "Kannada": (("kn", 0.96),),
    "Telugu": (("te", 0.96),),
    "Gujarati": (("gu", 0.96),),
    "Gurmukhi": (("pa", 0.96),),
    "Odia": (("or", 0.96),),
    "Meetei Mayek": (("mni", 0.94),),
    "Ol Chiki": (("sat", 0.94),),
    # Shared scripts remain intentionally less certain.
    "Bengali": (("bn", 0.50), ("as", 0.32), ("mni", 0.10)),
    "Devanagari": (
        ("hi", 0.34),
        ("mr", 0.18),
        ("ne", 0.12),
        ("sa", 0.10),
        ("mai", 0.08),
        ("kok", 0.07),
        ("doi", 0.06),
        ("brx", 0.05),
    ),
    "Arabic": (("ur", 0.50), ("ks", 0.25), ("sd", 0.18)),
    "Common": (("numeric", 0.99),),
    "Unknown": (("unknown", 0.0),),
}


def get_language(code: str) -> LanguageInfo:
    return SUPPORTED_LANGUAGES.get(code, SUPPORTED_LANGUAGES["unknown"])


def language_codes() -> tuple[str, ...]:
    return tuple(code for code in SUPPORTED_LANGUAGES if code not in {"mixed", "unknown", "numeric"})
