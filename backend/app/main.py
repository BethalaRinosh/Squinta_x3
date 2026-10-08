"""FastAPI application entry point."""

import logging
import os
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from app.config import settings
from app.database import Base, engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle hook.

    On startup we ensure required directories exist and create all DB
    tables (useful for local dev; in production you'd rely on Alembic
    migrations instead).
    """
    # Ensure storage directories exist.
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.MODEL_DIR, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Create database tables.
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Lightweight migrations for new columns on existing tables.
    from sqlalchemy import text

    async with engine.begin() as conn:
        try:
            await conn.execute(
                text("ALTER TABLE pages ADD COLUMN page_warped INTEGER NOT NULL DEFAULT 0")
            )
        except Exception:
            pass  # column already exists

    async with engine.begin() as conn:
        try:
            await conn.execute(
                text("ALTER TABLE ocr_results ADD COLUMN ink_layer VARCHAR(50) NOT NULL DEFAULT 'MAIN_INK'")
            )
        except Exception:
            pass

    async with engine.begin() as conn:
        try:
            await conn.execute(
                text("ALTER TABLE ocr_results ADD COLUMN ink_metadata TEXT")
            )
        except Exception:
            pass

    async with engine.begin() as conn:
        for column_sql in (
            "ALTER TABLE ocr_results ADD COLUMN language VARCHAR(20)",
            "ALTER TABLE ocr_results ADD COLUMN language_name VARCHAR(100)",
            "ALTER TABLE ocr_results ADD COLUMN script VARCHAR(50)",
            "ALTER TABLE ocr_results ADD COLUMN language_confidence FLOAT",
            "ALTER TABLE ocr_results ADD COLUMN translated_text TEXT",
            "ALTER TABLE ocr_results ADD COLUMN translation_applied INTEGER NOT NULL DEFAULT 0",
        ):
            try:
                await conn.execute(text(column_sql))
            except Exception:
                pass

    yield

    # Shutdown: dispose of the engine connection pool.
    await engine.dispose()


app = FastAPI(
    title="Squinta",
    description="Backend API for the Squinta web application.",
    version="0.1.0",
    lifespan=lifespan,
)

# ── Middleware ─────────────────────────────────────────────────────────────────

# Session middleware is required by authlib's Starlette integration for the
# OAuth state parameter.
app.add_middleware(
    SessionMiddleware,
    secret_key=settings.SECRET_KEY,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────

from app.routes.auth import router as auth_router  # noqa: E402
from app.routes.corrections import router as corrections_router  # noqa: E402
from app.routes.documents import router as documents_router  # noqa: E402
from app.routes.model import router as model_router  # noqa: E402
from app.routes.ocr import router as ocr_router  # noqa: E402
from app.routes.photos import router as photos_router  # noqa: E402
from app.routes.search import router as search_router  # noqa: E402

app.include_router(auth_router)
app.include_router(documents_router)
app.include_router(ocr_router)
app.include_router(corrections_router)
app.include_router(search_router)
app.include_router(photos_router)
app.include_router(model_router)

# ── Static files ──────────────────────────────────────────────────────────────
# Serve uploaded images so the frontend can display them directly.

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")

# ── Health check ──────────────────────────────────────────────────────────────


@app.get("/health", tags=["health"])
async def health_check() -> dict:
    return {"status": "ok"}
