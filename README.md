# Handwriting OCR

A web application that converts handwritten documents to text using OCR, then learns your handwriting over time through corrections. Upload or photograph handwritten pages (or import from Google Photos), get instant transcriptions via Gemini Flash or TrOCR, and correct mistakes through a gamified "Play" mode. Corrections feed back into per-user LoRA fine-tuning, improving accuracy over time.

## Features

- **Dual OCR Engine** -- Gemini 2.5 Flash API (primary, high-quality) with TrOCR local fallback
- **Auto-rotation** -- Detects and corrects image orientation (Gemini: single-prompt detection; TrOCR: tries all 4 orientations)
- **Perspective Warp** -- Detects notebook page corners in camera photos and applies perspective transform to remove desk/background, producing a clean rectangular page image
- **Deskew** -- Straightens small text skew via Hough line detection so bounding boxes align with horizontal text
- **Ink-Aware Bbox Alignment** -- Detects actual ink line positions using Otsu binarization + horizontal projection, then snaps Gemini's bounding boxes to real text positions (corrects spacing drift on long pages)
- **Auto-crop** -- Detects content bounds to focus OCR on the writing area
- **Custom Bounding Boxes** -- Draw boxes on the page to OCR specific regions
- **Bbox Training Mode** -- Manually redraw bounding boxes for any OCR result to correct alignment
- **Play Mode** -- Gamified correction interface that prioritizes low-confidence results
- **Personalized Fine-tuning** -- Per-user LoRA adapters trained on your corrections
- **Calibration** -- Bootstrap training with a single handwriting sample
- **Google Photos Import** -- Import photos directly via the Google Photos Picker API
- **Full-text Search** -- Whoosh-indexed search across all your transcribed documents
- **Model Export** -- Download your personalized LoRA weights

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- A Google Cloud project with OAuth 2.0 credentials
- (Optional) A [Gemini API key](https://aistudio.google.com/apikey) for high-quality OCR

### Setup

```bash
# Clone
git clone https://github.com/BethalaRinosh/Squinta_x3.git
cd Squinta_x3

# Configure
cp .env.example .env
# Edit .env with your Google OAuth credentials and (optional) Gemini API key

# Backend
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Frontend (in a separate terminal)
cd frontend
npm install
```

### Run

```bash
# Backend (from backend/)
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000

# Frontend (from frontend/)
npm run dev
```

Open http://localhost:5176 in your browser. The frontend dev server proxies API requests to the backend on port 8000.

### Docker

```bash
docker compose up --build
# App available at http://localhost:8000
```

## Recent Reliability Fixes

- Fixed uploaded image URLs on Windows so the UI uses `/api/uploads/...` instead of exposing a local `C:/Users/...` filesystem path.
- Persisted uploaded pages and their `processing` state before background OCR starts, preventing transaction/state races.
- Kept the document viewer polling OCR when a page is already processing when the viewer mounts.
- Removed unsupported Gemini thinking configuration from rotation and page-corner detection requests.

## Architecture

```
handwriting-ocr/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI app, CORS, lifespan, DB migration
│   │   ├── auth.py          # JWT token verification
│   │   ├── config.py        # Pydantic settings
│   │   ├── database.py      # SQLAlchemy async engine (SQLite w/ 30s busy timeout)
│   │   ├── models.py        # ORM models (User, Document, Page, OcrResult, Correction, UserModel)
│   │   ├── schemas.py       # Pydantic request/response schemas
│   │   ├── ocr.py           # OCR engines, image processing, bbox alignment
│   │   ├── finetune.py      # LoRA fine-tuning pipeline
│   │   └── routes/
│   │       ├── auth.py          # Google OAuth login/callback, JWT
│   │       ├── documents.py     # Upload, camera, CRUD, rotate, crop
│   │       ├── ocr.py           # Trigger OCR, get results, process bbox, train bbox
│   │       ├── corrections.py   # Submit corrections, Play mode batches
│   │       ├── search.py        # Full-text search (Whoosh)
│   │       ├── photos.py        # Google Photos Picker import
│   │       └── model.py         # Training, calibration, export
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── api.js               # Axios client + all API functions
│   │   ├── hooks/useAuth.jsx    # Auth context + token management
│   │   ├── components/
│   │   │   ├── PageViewer.jsx           # Image viewer with bbox overlays, draw/crop/train modes
│   │   │   └── BboxHighlightViewer.jsx  # Bbox drawing component (Play, Calibrate)
│   │   └── pages/
│   │       ├── Login.jsx         # Google OAuth sign-in
│   │       ├── Dashboard.jsx     # Document list, upload, navigation
│   │       ├── Upload.jsx        # Multi-source upload (file, camera, Google Photos)
│   │       ├── DocumentView.jsx  # Page viewer + OCR results + train mode
│   │       ├── Play.jsx          # Correction game with speech-to-text
│   │       ├── Search.jsx        # Full-text search
│   │       ├── Model.jsx         # Training controls + export
│   │       └── Calibrate.jsx     # Bootstrap training with sample text
│   └── package.json
├── Dockerfile              # Multi-stage (Node build + Python runtime)
├── docker-compose.yml
├── .env.example
└── CLAUDE.md               # AI assistant instructions
```

## How It Works

### Data Flow

1. **Upload** -- Images saved to `data/uploads/user_{id}/`, Document + Page rows created
2. **Pre-process** -- Auto-rotation detects orientation and bakes into the image file. Perspective warp detects notebook page corners and removes background. Deskew straightens small text skew.
3. **OCR** -- Gemini (or TrOCR) transcribes the page, returning text with bounding boxes. Ink line detection snaps bounding boxes to actual text positions.
4. **Correct** -- Play mode surfaces lowest-confidence results first, user approves or corrects
5. **Train** -- Corrections crop original images to bounding boxes, LoRA fine-tunes TrOCR decoder attention layers (q_proj, v_proj), new adapter version saved
6. **Improve** -- Subsequent OCR loads user's LoRA adapter for better accuracy

### OCR Pipeline

The full Gemini OCR pipeline for a page:

```
Camera Photo
    │
    ▼
┌─────────────────┐
│  Auto-Rotation   │  Gemini detects orientation → bake rotation into file
└────────┬────────┘
         ▼
┌─────────────────┐
│ Perspective Warp │  Gemini detects 4 page corners → OpenCV warpPerspective
└────────┬────────┘  removes desk/hands/background → clean page rectangle
         ▼
┌─────────────────┐
│     Deskew       │  Hough line detection → rotate to straighten text
└────────┬────────┘
         ▼
┌─────────────────┐
│   Gemini OCR     │  Full-page prompt → returns JSON array of
└────────┬────────┘  {text, box: [y1,x1,y2,x2]} entries (0-1000 normalized)
         ▼
┌─────────────────┐
│  Ink Detection   │  Otsu binarization → horizontal projection →
└────────┬────────┘  find actual text line positions in the image
         ▼
┌─────────────────┐
│  Bbox Alignment  │  Sequential matching: snap each Gemini text line
└────────┬────────┘  to nearest detected ink line (fixes spacing drift)
         ▼
    OCR Results
    (text + aligned bboxes)
```

**Gemini Flash** (when `GEMINI_API_KEY` is set):
- Sends full page image to Gemini 2.5 Flash for transcription with bounding boxes
- Auto-rotation via single-prompt orientation detection (0/90/180/270)
- Perspective warp uses Gemini to detect page corners, with 2% outward padding to avoid trimming content
- Bounding box alignment corrects Gemini's uniform y-spacing grid (which drifts from actual ruled-line spacing on notebook pages) by detecting real ink positions via Otsu binarization + horizontal projection
- API calls include automatic retry (3 attempts with exponential backoff) for transient network errors

**TrOCR** (local fallback):
- `microsoft/trocr-large-handwritten` via HuggingFace Transformers
- Line segmentation via horizontal projection profiles
- Auto-rotation by trying all 4 orientations, keeping highest confidence
- Confidence from average token log-probabilities

**Bounding Box Alignment** (`_build_direct_segments`):

Gemini returns text with bounding boxes that use a uniform vertical grid. On notebook pages with ruled lines, this grid spacing (~121px) often differs from the actual line spacing (~109px), causing progressive drift -- by line 25+, boxes can be 300px off from the actual text. The alignment pipeline fixes this:

1. **Ink line detection** (`_detect_ink_lines`): Otsu binarization on the middle 75% of the image width, horizontal projection to find rows with ink, contiguous run detection with minimum height filtering (15px), and automatic splitting of oversized blobs using projection valleys
2. **Body-start detection**: Identifies header/date lines (which have non-standard spacing to the next line) and excludes them from ink matching
3. **Sequential matching**: Each Gemini text line (in reading order) is matched to the next available ink line, skipping oversized blobs. This avoids the problem of distance-based matching where Gemini's drifted y-coordinates would match to the wrong ink line
4. **Extrapolation**: Unmatched lines at the end of the page are spaced using the median ink line spacing from the last matched position
5. **Height from ink**: Each box height comes from the actual ink line height, not a uniform value

### Fine-tuning

Per-user LoRA adapters via the PEFT library:
- **Rank:** 8, **Alpha:** 16
- **Target:** Decoder attention (q_proj, v_proj)
- **Storage:** `data/models/user_{id}/v{N}/`
- **Training:** CPU-compatible, small batches from correction pairs

## Configuration

Copy `.env.example` to `.env`:

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_CLIENT_ID` | Yes | Google Cloud OAuth 2.0 client ID |
| `GOOGLE_CLIENT_SECRET` | Yes | Google Cloud OAuth 2.0 client secret |
| `SECRET_KEY` | Yes | Random string for JWT signing |
| `GEMINI_API_KEY` | No | Gemini API key for high-quality OCR |
| `DATABASE_URL` | No | SQLite path (default: `sqlite:///./data/app.db`) |
| `UPLOAD_DIR` | No | Upload directory (default: `./data/uploads`) |
| `MODEL_DIR` | No | Model directory (default: `./data/models`) |

### Google Cloud Setup

1. Create a project in [Google Cloud Console](https://console.cloud.google.com/)
2. Enable the **Google Photos Picker API**
3. Create OAuth 2.0 credentials (Web application)
4. Set authorized redirect URI to `http://localhost:8000/auth/callback`
5. Copy client ID and secret to `.env`

## API Endpoints

### Authentication
| Method | Path | Description |
|--------|------|-------------|
| GET | `/auth/login` | Redirect to Google OAuth |
| GET | `/auth/callback` | OAuth callback, mints JWT |
| GET | `/auth/me` | Current user profile |
| POST | `/auth/logout` | Clear session |

### Documents
| Method | Path | Description |
|--------|------|-------------|
| GET | `/documents` | List all documents |
| POST | `/documents/upload` | Upload image files |
| POST | `/documents/camera` | Camera capture (base64) |
| GET | `/documents/{id}` | Document with pages + results |
| DELETE | `/documents/{id}` | Delete document and files |
| POST | `/documents/pages/{id}/rotate` | Rotate page image |
| POST | `/documents/pages/{id}/crop` | Set crop region |
| POST | `/documents/pages/{id}/crop/clear` | Clear crop |
| POST | `/documents/pages/{id}/crop/auto` | Auto-detect crop |

### OCR
| Method | Path | Description |
|--------|------|-------------|
| POST | `/ocr/process/{page_id}` | Trigger OCR (background) |
| POST | `/ocr/process-document/{doc_id}` | OCR all pages |
| POST | `/ocr/process-bbox/{page_id}` | OCR drawn region (sync) |
| GET | `/ocr/results/{page_id}` | Get OCR results |
| GET | `/ocr/processing-status` | Poll processing state |
| PUT | `/ocr/result/{result_id}/bbox` | Update result bounding box (train mode) |

### Corrections
| Method | Path | Description |
|--------|------|-------------|
| POST | `/corrections` | Submit correction |
| GET | `/corrections` | List corrections |
| GET | `/corrections/play` | Get Play mode batch |
| POST | `/corrections/play/submit` | Submit Play correction |

### Search, Photos, Model
| Method | Path | Description |
|--------|------|-------------|
| GET | `/search` | Full-text search |
| POST | `/photos/picker/session` | Create Photos Picker session |
| GET | `/photos/picker/session/{id}` | Poll picker status |
| POST | `/photos/picker/import` | Import selected photos |
| GET | `/model/status` | Model version + stats |
| POST | `/model/train` | Start fine-tuning |
| POST | `/model/calibrate` | Calibrate with sample |
| GET | `/model/export` | Download LoRA weights |

## Tech Stack

**Backend:** FastAPI, SQLAlchemy 2.0 (async), aiosqlite, python-jose (JWT), authlib (OAuth), OpenCV

**Frontend:** React 19, Vite 6, Tailwind CSS 4, React Router 6, TanStack Query 5, Axios

**ML/Vision:** Google GenAI (Gemini 2.5 Flash), Transformers (TrOCR), PEFT (LoRA), PyTorch, OpenCV (perspective warp, deskew, ink detection)

**Search:** Whoosh

## Deployment

Target: **Google Cloud Run** with persistent volume for `data/`.

The Dockerfile is multi-stage: Node builds the frontend, Python serves both the API and static assets. Mount `data/` as a persistent volume or use GCS for production storage.

## License

MIT
