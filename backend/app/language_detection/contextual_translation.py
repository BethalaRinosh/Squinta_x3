"""Context-aware translation helpers.

Protects names that context explicitly identifies as companies, brands, products,
or organizations while allowing ordinary uses of the same words to translate.
"""
from __future__ import annotations

import re

# Cues such as "My company's name is FISH", "our brand is Orange", and
# "The product is called Nothing".
_NAME_CUE_PATTERNS = (
    re.compile(
        r"\b(?:my|our|the)\s+(?:company|business|brand|startup|organisation|organization|"
        r"product|app|application|website|platform|team|project|channel)(?:['’]s)?\s+"
        r"(?:name\s+is\s+|is\s+called\s+|is\s+named\s+|is\s+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:my|our|the)\s+(?:company|business|brand|startup|organisation|organization|"
        r"product|app|application|website|platform|team|project|channel)['’]s\s+name\s+is\s+",
        re.IGNORECASE,
    ),
    re.compile(
    re.compile(
        r"\\b(?:my|our|the)\\s+(?:company|business|brand|startup|organisation|organization|"
        r"product|app|application|website|platform|team|project|channel)[\'’]s\\s+name\\s+is\\s+"
        re.IGNORECASE,
    ),
        r"\b(?:my\s+)?(?:company|business|brand|startup|organisation|organization|"
        r"product|app|application|website|platform|team|project|channel)\s+name\s+is\s+",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:company|brand|product|app|platform|organization|organisation)\s+"
        r"(?:is\s+called|is\s+named|called|named)\s+",
        re.IGNORECASE,
    ),
)
_ORG_SUFFIX = re.compile(
    r"(?P<name>[A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*){0,4})"
    r"\s+(?:Inc\.?|LLC|Ltd\.?|Limited|Corporation|Corp\.?|Technologies|"
    r"Technology|Labs|Studio|Studios|University|Foundation)\b"
)
_STOP = re.compile(r"[,;:!?\n\r]")
_TRAILING_PUNCTUATION = " \t\r\n,;:!?)]}>."

def find_protected_names(text: str) -> list[str]:
    """Return names supported by explicit context, not capitalization alone."""
    if not text:
        return []
    candidates: list[tuple[int, int, str]] = []
    for cue in _NAME_CUE_PATTERNS:
        for match in cue.finditer(text):
            start = match.end()
            remainder = text[start:]
            stop = _STOP.search(remainder)
            raw = remainder[:stop.start()] if stop else remainder
            name = raw.strip().rstrip(".").strip()
            if name:
                end = start + len(raw.rstrip(_TRAILING_PUNCTUATION + "."))
                candidates.append((start, end, name))
    for match in _ORG_SUFFIX.finditer(text):
        name = match.group("name").strip()
        if name:
            candidates.append((match.start("name"), match.end("name"), name))
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
        masked, count = re.subn(r"(?<!\w)" + re.escape(name) + r"(?!\w)", placeholder, masked, flags=re.IGNORECASE)
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
    """Translate a sentence while keeping contextually identified names unchanged.

    ``translator`` is a callback with the same arguments as ``translate_text``.
    """
    names = find_protected_names(text)
    if not names:
        return translator(text, source_language, target_language)
    masked, replacements = _mask_names(text, names)
    translated = translator(masked, source_language, target_language)
    return _restore_names(translated or "", replacements)
