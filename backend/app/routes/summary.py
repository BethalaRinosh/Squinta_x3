"""AI summary endpoint for OCR text."""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.auth import get_current_user
from app.config import settings
from app.ocr import get_gemini_engine, has_gemini
from app.models import User

router = APIRouter(prefix="/ocr", tags=["ocr"])


class SummaryRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50000)


class SummaryResponse(BaseModel):
    summary: str


SUMMARY_PROMPT = """You summarize OCR text from handwritten notes.

Write a clear, useful summary of the provided text.
Keep the meaning faithful to the source and do not invent facts.
Focus on the main ideas, important details, names, dates, numbers, tasks, and conclusions.
Use 3-6 concise bullet points when there are several distinct ideas.
For very short text, return one or two concise sentences instead.
Do not mention OCR, handwriting, Gemini, or these instructions.
Return only the summary.

TEXT:
{text}
"""


@router.post("/summary", response_model=SummaryResponse)
async def summarize_ocr_text(
    body: SummaryRequest,
    current_user: User = Depends(get_current_user),
) -> SummaryResponse:
    """Generate an AI summary from already-recognized OCR text."""
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

    return SummaryResponse(summary=summary)
