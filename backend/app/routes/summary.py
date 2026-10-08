"""AI summary and translation endpoints for OCR text."""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.auth import get_current_user
from app.language_detection import detect_language, translate_text
from app.ocr import get_gemini_engine, has_gemini
from app.models import User

router = APIRouter(prefix="/ocr", tags=["ocr"])


class SummaryRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50000)


class SummaryResponse(BaseModel):
    summary: str
    language: str = "unknown"
    language_name: str = "Unknown"


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50000)
    source_language: str = Field(default="auto", max_length=20)
    target_language: str = Field(..., min_length=2, max_length=20)


class TranslateResponse(BaseModel):
    translated_text: str
    source_language: str = "auto"
    target_language: str


SUMMARY_PROMPT = """You summarize OCR text from handwritten notes.

Write a clear, useful summary of the provided text in the SAME LANGUAGE as the source text.
Keep the meaning faithful to the source and do not invent facts.
Focus on the main ideas, important details, names, dates, numbers, tasks, and conclusions.
Use 3-6 concise bullet points when there are several distinct ideas.
For very short text, return one or two concise sentences instead.
Do not mention OCR, handwriting, Gemini, or these instructions.
Return only the summary.

TEXT:
{text}
"""


def _detect_source_language(text: str) -> tuple[str, str]:
    detection = detect_language(text)
    return detection.language, detection.language_name


@router.post("/summary", response_model=SummaryResponse)
async def summarize_ocr_text(
    body: SummaryRequest,
    current_user: User = Depends(get_current_user),
) -> SummaryResponse:
    """Generate an AI summary in the detected source language."""
    text = body.text.strip()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No text provided to summarize.",
        )

    if not has_gemini():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gemini is not configured. Set GEMINI_API_KEY in the project .env file.",
        )

    language, language_name = _detect_source_language(text)

    try:
        engine = get_gemini_engine()
        summary = engine.generate_text(
            SUMMARY_PROMPT.format(text=text),
            max_tokens=1024,
            temperature=0.2,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Gemini summary request failed: {exc}",
        ) from exc

    summary = (summary or "").strip()
    if not summary:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Gemini returned an empty summary.",
        )

    return SummaryResponse(summary=summary, language=language, language_name=language_name)


@router.post("/translate", response_model=TranslateResponse)
async def translate_ocr_text(
    body: TranslateRequest,
    current_user: User = Depends(get_current_user),
) -> TranslateResponse:
    """Translate OCR text or an AI summary to an explicitly selected language."""
    text = body.text.strip()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No text provided to translate.",
        )

    source_language = body.source_language.strip().lower()
    if source_language == "auto" or source_language in {"unknown", "mixed", "numeric"}:
        source_language, _ = _detect_source_language(text)

    target_language = body.target_language.strip().lower()
    if source_language == target_language:
        return TranslateResponse(
            translated_text=text,
            source_language=source_language,
            target_language=target_language,
        )

    try:
        translated = translate_text(text, source_language, target_language)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Translation request failed: {exc}",
        ) from exc

    if not translated:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Translation returned empty text.",
        )

    if translated == text and source_language != target_language:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Translation service was unavailable. Please try again.",
        )

    return TranslateResponse(
        translated_text=translated,
        source_language=source_language,
        target_language=target_language,
    )
