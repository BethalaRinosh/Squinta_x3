from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# .env lives at the project root (one level above backend/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_env_file = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=str(_env_file),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Google OAuth2
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""

    # App origins
    BACKEND_URL: str = "http://localhost:8000"
    FRONTEND_URL: str = "http://localhost:5176"

    # JWT / session secret
    SECRET_KEY: str = "change-me-in-production"

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/app.db"

    # File storage
    UPLOAD_DIR: str = "./data/uploads"
    MODEL_DIR: str = "./data/models"

    # Gemini API (for high-quality OCR via multimodal LLM)
    GEMINI_API_KEY: str = ""
    # Gemini model used by the OCR pipeline. Keep this configurable because
    # model availability can change independently of the application code.
    GEMINI_MODEL: str = "gemini-3.5-flash-lite"

    # Conservative image preprocessing for Gemini OCR. Gemini is already
    # rotation-aware, and mutating the stored upload can make the UI display
    # a rotated or stretched image when a vision model misreads the page.
    # Keep these off by default; they can be enabled explicitly for testing.
    GEMINI_AUTO_ROTATE: bool = False
    GEMINI_PAGE_WARP: bool = False
    GEMINI_DESKEW: bool = False

    # Optional OpenAI OCR provider (Responses API; useful for handwritten text).
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-5.6"

    # Allow the optional TrOCR fallback only when it is already cached locally
    # and the operator has explicitly opted in to a potentially large model download.
    ENABLE_TROCR_FALLBACK: bool = False

    # Enable domain-aware OCR verification.
    ENABLE_CONTEXT_ENGINE: bool = True

    # JWT config
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days

    @field_validator("UPLOAD_DIR", "MODEL_DIR", mode="before")
    @classmethod
    def resolve_relative_storage_paths(cls, value: str | None) -> str | None:
        """Anchor relative storage paths to the project root.

        The app may be launched from the backend folder, which would otherwise
        cause uploads and model files to be saved under backend/data instead of
        the workspace data directory.
        """
        if value is None:
            return value

        path = Path(value)
        if not path.is_absolute():
            path = (_PROJECT_ROOT / path).resolve()
        return str(path)


settings = Settings()
