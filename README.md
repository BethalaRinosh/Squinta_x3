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
- **Text-to-Speech** -- Select OCR text and read it aloud, or read individual results, using the browser's built-in speech synthesis
- **AI Summary** -- Generate a concise Gemini summary of the current page's OCR text without sending the image through another vision pass, with one-click read-aloud playback using the existing browser speech synthesis
- **Strike-off detection** -- Detects text crossed out by a strong horizontal stroke. In Document View, **Remove strike-offs** hides detected struck-through text; when disabled, it remains visible with a strike-through.
- **Model Export** -- Download your personalized LoRA weights
- **Domain Context Engine** -- Detects likely medical, legal, finance, science, or education context from OCR candidates and runs a constrained Gemini verification pass using domain terminology, without inventing unsupported text.
- **Context-aware translation** -- Uses explicit naming context (for example, “My company's name is FISH”) to protect company, brand, and product names across translation providers, while allowing ordinary words such as “fish” to translate normally when they are not identified as names.
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
- Fixed Docker's backend module path and data-volume paths.
- Docker now serves the built React frontend from the FastAPI app, while keeping the `/health` endpoint reachable.
- Removed a credential value from `.env.example`. If the old Gemini key was real, revoke/rotate it in Google AI Studio because removing it from the latest file does not erase it from Git history.

### Domain-aware OCR context

The context engine runs after the first Gemini handwriting pass. It detects a likely domain from the candidate transcription, injects a compact terminology/context profile, and asks Gemini to verify ambiguous readings against the original image. The context layer is advisory only: it cannot create text that is not visually supported. Medical context includes common clinical terms, drugs, measurements, abbreviations, and units. The engine also has legal, finance, science, and education profiles and can be extended with larger ontologies such as UMLS/MeSH/RxNorm through `backend/app/context_engine.py`.

Set `ENABLE_CONTEXT_ENGINE=false` to disable the second-pass refinement.

- **Summary speech playback** -- AI summaries can now be read aloud directly from the summary card, with the existing speech-rate controls and stop/play behavior shared by OCR text-to-speech.

### Gemini OCR reliability

Gemini OCR requests are bounded by a client-side timeout, and structured OCR output is capped at a practical size with JSON response mode enabled. This prevents a stalled vision request from leaving a page in `processing` indefinitely. Visual-only detections are also preserved when a page contains structures but no recognized text.

### Visual-element API compatibility

The visual-structure persistence layer now maps the SQLAlchemy `element_metadata` attribute to the public `metadata` API field correctly, and page endpoints eagerly load visual elements to avoid async lazy-loading failures.

## Audit Status

A source-level audit has been completed across the backend, frontend, OCR pipeline, configuration, authentication, storage, Docker setup, and repository hygiene. No remaining occurrences were found for the known branding mismatch, leaked key pattern, unsupported Gemini thinking configuration, Windows filesystem URL pattern, or invalid FastAPI background-task default.

The repository still needs a real local validation pass with your installed Python/Node environments. The GitHub review environment can inspect and patch source, but it cannot execute your local OCR stack, browser, Docker daemon, or external Google/OpenAI credentials.

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
│   │       └── summary.py       # Generate AI summaries from OCR text
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
- Sends the full page image to the configured Gemini Flash model (default `gemini-3.5-flash-lite`) for transcription with bounding boxes
- Auto-rotation via single-prompt orientation detection (0/90/180/270)
- Perspective warp uses Gemini to detect page corners, with 2% outward padding to avoid trimming content
- Bounding box alignment corrects Gemini's uniform y-spacing grid (which drifts from actual ruled-line spacing on notebook pages) by detecting real ink positions via Otsu binarization + horizontal projection
- Gemini requests use an explicit API key, a bounded client timeout, and automatic retry for transient failures. The API key is never taken from Google OAuth credentials.

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
| POST | `/ocr/summary` | Generate an AI summary from OCR text using Gemini |

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

**Frontend:** React 19, Vite 6, Tailwind CSS 4, React Router 6, TanStack Query 5, Axios, Web Speech API (browser-native text-to-speech)

**ML/Vision:** Google GenAI (Gemini 2.5 Flash), Transformers (TrOCR), PEFT (LoRA), PyTorch, OpenCV (perspective warp, deskew, ink detection)

**Search:** Whoosh

## Deployment

Target: **Google Cloud Run** with persistent volume for `data/`.

The Dockerfile is multi-stage: Node builds the frontend, Python serves both the API and static assets. Mount `data/` as a persistent volume or use GCS for production storage.

## License

MIT


### Latest Reliability Fix

- Fixed FastAPI route parameter-ordering errors in the upload and model-training endpoints so the backend can import successfully under Python 3.11.


### Authentication reliability

- Google OAuth browser navigation now targets the configured backend origin directly instead of depending on Vite's `/api` proxy. This makes local sign-in reliable at `http://localhost:5176` while keeping the backend callback at `http://localhost:8000/auth/callback`.
- Set `VITE_BACKEND_URL` when the frontend and backend use different origins.

### Visual structure recognition

Squinta now asks the vision OCR pipeline to detect meaningful non-text handwriting structures alongside text, including arrows, brackets, tables/grids, boxes, circles, underlines, connectors, and larger diagrams. These are stored as structured page elements and rendered separately in the document viewer, so symbols are not forced into the OCR text stream.
\n### Visual geometry alignment

Visual-element overlays now use the detected geometry rather than drawing generic CSS rectangles or horizontal arrows. Table geometry is now recovered independently from long page-level strokes when Gemini's table bbox is displaced, and the viewer only draws divider lines that are actually detected in the image instead of manufacturing a regular grid. Gemini is prompted to return ordered points for arrows, connectors, underlines, brackets, boxes, circles, and tables; the backend converts those points to image-pixel coordinates and conservatively refines strong straight edges/quadrilaterals with OpenCV. The frontend renders the resulting paths, skewed outlines, ellipses, and perspective-aware table grids in an SVG overlay. Legacy normalized geometry remains supported.


### OCR visual modes

OCR now defaults to a fast mode that focuses on handwritten text, text positions, and arrows. The Document View includes a **Detailed visuals** toggle. Turn it on before processing when you need tables, boxes, circles, brackets, underlines, connectors, and diagrams. The setting is remembered in the browser, while new uploads default to the faster mode.


### Position-aware formatted OCR
The document OCR panel now includes a **Formatted** view that reconstructs line breaks, horizontal spacing, indentation, and rough column structure from each OCR result's bounding-box position. The original result list remains available under **Results**. This is a layout reconstruction layer, not a claim of exact table/diagram recovery.

The Formatted view also preserves detected table structure: table borders and detected internal row/column dividers are rendered around OCR text assigned to their corresponding cells. Exact table geometry still depends on the optional Detailed visuals detection pass.


### Frontend build fix
Fixed malformed escaped template literals in the Formatted/Results OCR view toggle that caused Vite/Babel to fail parsing `DocumentView.jsx`.


### 2D formatted document reconstruction
The Formatted OCR view now reconstructs the page in 2D instead of flattening OCR into a text block. OCR text uses its original pixel bounding boxes, while detected arrows, table outlines/grid lines, boxes, circles, connectors, brackets, and underlines are rendered from their stored geometry. This keeps the formatted output spatially faithful to the source page.


## OCR image safety

Gemini OCR now keeps the original uploaded image unchanged by default. Automatic rotation, perspective page warping, and deskewing are disabled because a bad vision-model geometry/orientation guess can make the displayed image appear sideways or stretched and can cause OCR to lose the handwriting.

The controls are available as explicit backend environment flags for controlled testing:

- `GEMINI_AUTO_ROTATE=false`
- `GEMINI_PAGE_WARP=false`
- `GEMINI_DESKEW=false`

Gemini OCR bounding boxes are therefore generated against the same pixel space that the browser displays. The JSON parser also accepts common structured-response wrappers such as `results`, `items`, and `detections` instead of silently dropping a valid response.

### AI Summary

After OCR finishes, the Document View shows an **AI Summary** button in the OCR Results panel. It sends the recognized page text to the same configured Gemini model used by Squinta and returns a concise summary. The image is not sent again for this feature, so it uses a regular text-generation request rather than another vision OCR pass.

The feature uses the existing `GEMINI_API_KEY` and `GEMINI_MODEL` settings. No additional API key or dependency is required.

## Stability pass

The current build includes a focused reliability pass covering frontend search highlighting, Google Photos background OCR transaction ordering, and OCR processing-status polling. Existing OCR, translation, summaries, speech playback, correction, training, and visual-structure features are preserved.
\n\n### Multilingual translation controls

- OCR preserves the recognized source-language text instead of silently replacing it with English. For non-English OCR results, the Results view shows a **Translate** control with a target-language selector.
- AI summaries are generated in the detected source language, and the summary card provides its own **Translate** control when the source language is not English.
- Translation uses the backend `deep-translator` integration and reports a clear error when the translation service is unavailable.

### OCR confidence scoring\n\nOCR confidence is now evidence-based instead of a fixed 90%/95% value. The displayed score combines Gemini's confidence only as a weak prior with observable transcription evidence such as output completeness, visible ink density, and explicit uncertainty markers. Scores remain normalized to 0-100% for the existing UI, but can now vary with the quality of the recognized region.\n

### Strike-through detection and OCR panel layout

- Strike-off detection now checks approximate word-level boxes when the OCR engine returns a whole line as one box, so hiding a crossed-out word can preserve the surrounding words.
- The OCR Results header controls wrap on narrow panels instead of overflowing or clipping.
- Re-process the page after pulling this update so stored OCR results include the improved word-level strike-off metadata.


### Context-aware translation name preservation

Translation now detects names supported by context, including names that are ordinary words (such as **FISH** in “My company's name is FISH”). It replaces those names with temporary placeholders before translation and restores the exact original spelling afterward. This runs before the provider selection, so it applies to both Google Translate and the Gemini fallback. Ordinary uses such as “I like fish” are not frozen and continue through the existing translation path.

The detection is intentionally conservative: it uses explicit naming phrases and common organization suffixes rather than protecting every capitalized word. Run `pytest backend/tests/test_contextual_translation.py` from the repository root to check the core behavior.

### Translation rate-limit resilience

- Successful translations are cached to avoid duplicate requests for the same text and language pair.
- If the unofficial Google Translate endpoint rate-limits a request, Squinta falls back to the configured Gemini model when `GEMINI_API_KEY` is available.
- Without a Gemini API key, the UI receives a clear temporary rate-limit message instead of a long raw provider error. Restart the backend after pulling this change.


### OCR confidence scoring correction

- Fixed a Gemini structured-page OCR path that assigned every recognized line a hard-coded `95%`, bypassing the confidence estimator.
- The displayed percentage is now a **heuristic reliability score**, based on the model prior, transcription length, visible ink in the OCR crop, expected ink support, and explicit uncertainty markers.
- Blank or nearly blank crops are capped at a low score. This score is not a statistically calibrated probability that every character is correct; calibration against a labeled test set is required before interpreting it as exact accuracy.
- Re-process existing pages to replace previously stored confidence values. Run `pytest backend/tests/test_ocr_confidence.py` from the repository root to test the scoring behavior.
