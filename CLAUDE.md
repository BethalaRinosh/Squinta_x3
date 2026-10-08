# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Handwriting OCR web app with personalized model fine-tuning. Users sign in with Google, upload/photograph handwritten pages (or import from Google Photos), get OCR results via TrOCR, then correct errors through a "Play" mode that prioritizes uncertain predictions. Corrections feed back into per-user LoRA fine-tuning, improving accuracy over time. Models are exportable.

## Commands

### Backend (from repo root)
```bash
# Setup
cd backend && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run dev server
uvicorn backend.app.main:app --reload --port 8000

# Type check
cd backend && python -m py_compile app/main.py
```

### Frontend (from repo root)
```bash
cd frontend && npm install
npm run dev          # Dev server on :5173, proxies /api to :8000
npm run build        # Production build to frontend/dist/
```

### Docker
```bash
docker compose up --build    # Full stack on :8000
```

## Architecture

**Backend:** FastAPI (async) with SQLAlchemy + aiosqlite (SQLite). JWT auth via python-jose, Google OAuth via authlib.

**Frontend:** React 18 + Vite + Tailwind CSS v4. React Router v6, TanStack Query for data fetching, Axios for API calls.

**OCR Engine:** `backend/app/ocr.py` — TrOCR (`microsoft/trocr-large-handwritten`) via HuggingFace transformers. Singleton `OcrEngine` that lazy-loads the model. Line segmentation via horizontal projection profiles. Confidence from average token log-probabilities.

**Fine-tuning:** `backend/app/finetune.py` — LoRA adapters (r=8, alpha=16) on decoder attention (q_proj, v_proj) via PEFT library. Per-user versioned weights stored in `data/models/user_{id}/v{N}/`. Training runs on CPU with small batches.

**Search:** Whoosh full-text index, built/updated as OCR results are created. Index stored at `data/whoosh_index/`.

### Key data flow
1. Image uploaded → saved to `data/uploads/user_{id}/` → Document + Page rows created
2. OCR triggered → `OcrEngine.process_page()` segments lines, runs TrOCR, stores OcrResult rows with confidence + bounding boxes → indexed in Whoosh
3. Play mode → backend returns lowest-confidence uncorrected results → user corrects → Correction rows created
4. Train → `FineTuner.train()` loads corrections, crops images to bboxes, LoRA fine-tunes TrOCR → new adapter version saved
5. Subsequent OCR loads user's LoRA adapter for better results

### Backend route modules (all in `backend/app/routes/`)
- `auth.py` — Google OAuth login/callback, JWT minting, `/auth/me`
- `documents.py` — CRUD, file upload, camera capture
- `ocr.py` — trigger OCR processing (background tasks), get results
- `corrections.py` — submit corrections, Play mode prioritized batches
- `search.py` — Whoosh full-text search across user's OCR results
- `photos.py` — Google Photos album browsing and import
- `model.py` — training status, trigger fine-tuning, export LoRA weights

### Frontend pages (all in `frontend/src/pages/`)
- `Login.jsx` — Google sign-in
- `Dashboard.jsx` — document list, upload/camera/photos entry points
- `Upload.jsx` — drag-drop upload, camera capture, Google Photos browser
- `DocumentView.jsx` — page image with OCR bbox overlays + results panel
- `Search.jsx` — debounced full-text search with highlighted matches
- `Play.jsx` — correction game (image + text + speech-to-text input)
- `Model.jsx` — training controls + model export

## Configuration

Copy `.env.example` to `.env` and fill in:
- `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` — from Google Cloud Console (enable Google Photos Library API, set OAuth redirect to `http://localhost:8000/auth/callback`)
- `SECRET_KEY` — random string for JWT signing

## Deployment

Target: Google Cloud Run. The Dockerfile is multi-stage (Node build → Python runtime). Uploaded images and models are stored in `data/` — mount a persistent volume or use GCS for production.
