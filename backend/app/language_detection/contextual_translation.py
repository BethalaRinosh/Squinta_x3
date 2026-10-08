"""Context-aware translation helpers.

Protects likely proper nouns and explicitly declared names while translating the
surrounding sentence. Common words such as "fish" are not protected unless context
marks them as a name/brand.
"""
from __future__ import annotations

import re

# Explicit semantic cues let us preserve names which are ordinary words too,
# e.g. "My company's name is FISH". Generic capitalisation alone is not enough.
_NAME_CUE_PATTERNS = (
    re.compile(
        r"\b(?:my\s+)?(?:company|business|brand|startup|organisation|organization|"
        r"product|app|application|website|platform|team|project|channel|company\'s\s+name|brand\'s\s+name)"
        r"\s+(?:is|is\s+called|named|called)\s+"
        r"(?P<name>[\w][\w&.-]*(?:\s+[\w][\w&.-]*){0,4})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:the\s+)?(?:company|brand|product|app|platform|organization|organisation)"
        r"\s+(?:name\s+is|is\s+called|called|named)\s+"
        r"(?P<name>[\w][\w&.-]*(?:\s+[\w][\w&.-]*){0,4})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<name>[A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*){0,4})"
        r"\s+(?:Inc\.?|LLC|Ltd\.?|Limited|Corporation|Corp\.?|Technologies|Technology|Labs|Studio|Studios|University|Foundation)\b"
    ),
)

_TRAILING_PUNCTUATION = " \t\r\n,;:!?)]}>."

def find_protected_names(text: str) -> list[str]:
    """Find names supported by explicit naming context or organization suffixes."""
    if not text:
        return []
    candidates: list[tuple[int, int, str]] = []
    for pattern in _NAME_CUE_PATTERNS:
        for match in pattern.finditer(text):
            raw = match.group("name")
            name = raw.rstrip(_TRAILING_PUNCTUATION).strip()
            if name:
                start = match.start("name")
                candidates.append((start, start + len(name), name))
    candidates.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    result: list[str] = []
    occupied: list[tuple[int, int]] = []
    for start, end, name in candidates:
        if any(start < used_end and end > used_start for used_start, used_end in occupied):
            continue
        occupied.append((start, end))
        if name not in result:
            result.append(name)
    return result

def _mask_names(text: str, names: list[str]) -> tuple[str, dict[str, str]]:
    """Replace protected name occurrences with stable placeholders, longest first."""
    masked = text
    replacements: dict[str, str] = {}
    for index, name in enumerate(sorted(names, key=len, reverse=True)):
        placeholder = f"ZXQPROTECTEDNAME{index}QXZ"
        masked, count = re.subn(re.escape(name), placeholder, masked, flags=re.IGNORECASE)
        if count:
            replacements[placeholder] = name
    return masked, replacements

def _restore_names(text: str, replacements: dict[str, str]) -> str:
    """Restore name placeholders, tolerating provider changes to placeholder casing."""
    restored = text
    for placeholder, original in replacements.items():
        restored = re.sub(re.escape(placeholder), lambda _: original, restored, flags=re.IGNORECASE)
    return restored

def translate_with_context(text: str, source_language: str, target_language: str, translator) -> str:
    """Translate the sentence while keeping contextually identified names unchanged.

    ``translator`` is a callback with the same arguments as ``translate_text``.
    """
    names = find_protected_names(text)
    if not names:
        return translator(text, source_language, target_language)
    masked, replacements = _mask_names(text, names)
    translated = translator(masked, source_language, target_language)
    return _restore_names(translated or "", replacements)
