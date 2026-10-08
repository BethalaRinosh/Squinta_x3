"""
OCR inference pipeline for handwriting OCR.

Primary engine: Google Gemini Flash (multimodal LLM) — dramatically better
accuracy on camera photos of handwriting, handles rotation natively.
Fallback engine: TrOCR-large-handwritten with optional LoRA adapters.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from huggingface_hub import try_to_load_from_cache
from PIL import Image, ImageFilter
from pillow_heif import register_heif_opener
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Enable HEIC/HEIF support in Pillow
register_heif_opener()

from app.config import settings

logger = logging.getLogger(__name__)

BASE_MODEL_NAME = "microsoft/trocr-large-handwritten"


def should_try_trocr_fallback() -> bool:
    """Return True only when the optional TrOCR fallback is explicitly enabled
    and the model is already cached locally.

    This avoids a surprise multi-gigabyte download on machines that only have
    the Gemini path available.
    """
    if not settings.ENABLE_TROCR_FALLBACK:
        logger.info("TrOCR fallback disabled by configuration.")
        return False

    candidate_files = [
        "config.json",
        "preprocessor_config.json",
        "pytorch_model.bin",
    ]
    for filename in candidate_files:
        try:
            if try_to_load_from_cache(repo_id=BASE_MODEL_NAME, filename=filename) is not None:
                return True
        except Exception:
            logger.warning("Unable to check local TrOCR cache for %s", filename, exc_info=True)

    logger.warning(
        "TrOCR fallback is enabled but the model is not cached locally; "
        "skipping download to avoid a network-timeout failure."
    )
    return False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class OcrSegment:
    """A single recognised text segment with its position on the page."""

    text: str
    confidence: float
    bbox: tuple[int, int, int, int]  # (x, y, w, h) relative to original image


@dataclass
class VisualElement:
    """A non-text visual structure detected on the page."""
    element_type: str
    confidence: float
    bbox: tuple[int, int, int, int]
    label: str | None = None
    geometry: str | None = None
    metadata: str | None = None


# ---------------------------------------------------------------------------
# Image preprocessing helpers
# ---------------------------------------------------------------------------


def preprocess_image(image_path: str | Path, rotation: int = 0) -> Image.Image:
    """Load an image from disk, convert to RGB, and apply manual rotation."""
    image_path = os.path.normpath(str(image_path).replace("\\", os.sep))
    img = Image.open(image_path)
    # EXIF rotation is now baked into the file at upload time,
    # so we only need to handle manual user rotation here.
    if img.mode != "RGB":
        img = img.convert("RGB")
    if rotation:
        # PIL rotate is counter-clockwise; user rotation is clockwise
        img = img.rotate(-rotation, expand=True)
    return img


def binarize(image: Image.Image) -> Image.Image:
    """Adaptive binarization for cleaner OCR input.

    Converts to grayscale, applies a local-mean adaptive threshold
    (approximated with a large-radius Gaussian blur), then binarizes.
    Uses morphological opening to remove noise from camera photos.
    """
    gray = image.convert("L")

    # Use a blurred version of the image as the local threshold.
    # A large radius approximates an adaptive neighbourhood.
    blur_radius = max(gray.width, gray.height) // 30
    blur_radius = max(blur_radius, 10)
    blurred = gray.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    gray_arr = np.array(gray, dtype=np.float32)
    blur_arr = np.array(blurred, dtype=np.float32)

    # Pixel is foreground (black) when it is darker than the local mean
    # minus offset.  Use a generous offset (25) so camera noise, paper
    # texture, and shadows don't get classified as ink.
    offset = 25.0
    binary = ((gray_arr < blur_arr - offset) * 255).astype(np.uint8)

    # Morphological opening (erosion → dilation) removes small noise specks
    # while preserving text strokes.  Use a 3×3 kernel.
    from PIL import ImageMorph

    binary_img = Image.fromarray(binary, mode="L")

    # Use min/max filters as a simpler morphological open.
    # Erosion = min filter, Dilation = max filter.
    kernel_size = max(3, min(image.width, image.height) // 500)
    if kernel_size % 2 == 0:
        kernel_size += 1
    binary_img = binary_img.filter(ImageFilter.MinFilter(kernel_size))
    binary_img = binary_img.filter(ImageFilter.MaxFilter(kernel_size))

    return binary_img


def _horizontal_projection(binary_image: Image.Image) -> np.ndarray:
    """Compute horizontal projection profile (sum of white pixels per row)."""
    arr = np.array(binary_image, dtype=np.float32) / 255.0
    return arr.sum(axis=1)


def segment_lines(image: Image.Image) -> list[tuple[Image.Image, tuple[int, int, int, int]]]:
    """Segment a page image into individual text lines.

    Uses horizontal projection profile on a binarized copy to find gaps
    between text lines.  Returns a list of ``(line_image, bbox)`` tuples
    where *line_image* is cropped from the **original** (RGB) image and
    *bbox* is ``(x, y, w, h)``.
    """
    binary = binarize(image)
    projection = _horizontal_projection(binary)

    # Threshold: a row is "text" if projection exceeds a fraction of the
    # maximum projection value.
    threshold = max(projection.max() * 0.02, 1.0)

    in_text = False
    line_starts: list[int] = []
    line_ends: list[int] = []

    for row_idx, val in enumerate(projection):
        if not in_text and val >= threshold:
            in_text = True
            line_starts.append(row_idx)
        elif in_text and val < threshold:
            in_text = False
            line_ends.append(row_idx)

    # Close the last region if the image ends while still in text.
    if in_text:
        line_ends.append(len(projection))

    # Merge lines that are very close together (likely the same line with
    # a thin gap, e.g. from descenders/ascenders).
    min_gap = max(int(image.height * 0.005), 3)
    merged_starts: list[int] = []
    merged_ends: list[int] = []

    for start, end in zip(line_starts, line_ends):
        if merged_ends and (start - merged_ends[-1]) < min_gap:
            # Extend the previous region.
            merged_ends[-1] = end
        else:
            merged_starts.append(start)
            merged_ends.append(end)

    # Add a small vertical padding around each line for better recognition.
    pad = max(int(image.height * 0.005), 2)

    results: list[tuple[Image.Image, tuple[int, int, int, int]]] = []
    for start, end in zip(merged_starts, merged_ends):
        y0 = max(start - pad, 0)
        y1 = min(end + pad, image.height)
        # Full width of the page.
        bbox = (0, y0, image.width, y1 - y0)
        line_img = image.crop((0, y0, image.width, y1))
        results.append((line_img, bbox))

    # If segmentation found nothing, return the whole image as one segment.
    if not results:
        results.append((image.copy(), (0, 0, image.width, image.height)))

    # Fallback: if we have a single segment covering most of the page,
    # the binarization likely failed to find gaps (common on camera photos).
    # Split into estimated line-height strips so TrOCR gets manageable chunks.
    if len(results) == 1:
        seg_img, seg_bbox = results[0]
        seg_height = seg_bbox[3]
        if seg_height > image.height * 0.6:
            # Estimate line height as ~3-4% of page height (typical for
            # handwriting on letter-size paper photographed at full res).
            est_line_height = max(int(image.height * 0.035), 40)
            results = []
            y = 0
            while y < image.height:
                y1 = min(y + est_line_height, image.height)
                bbox = (0, y, image.width, y1 - y)
                line_img = image.crop((0, y, image.width, y1))
                results.append((line_img, bbox))
                y = y1

    return results


def detect_content_bounds(
    image: Image.Image, padding_percent: float = 0.03,
) -> tuple[int, int, int, int] | None:
    """Detect the bounding box of actual content (ink) in an image.

    Uses horizontal and vertical projection profiles on a binarised copy to
    find where ink exists, trimming empty margins/background.

    Returns ``(x, y, w, h)`` with padding, or ``None`` if the detected
    region covers almost the entire image (no meaningful crop).
    """
    binary = binarize(image)
    arr = np.array(binary, dtype=np.float32) / 255.0

    h_proj = arr.sum(axis=1)         # per-row ink
    v_proj = arr.sum(axis=0)         # per-column ink

    h_thresh = max(h_proj.max() * 0.01, 1.0)
    v_thresh = max(v_proj.max() * 0.01, 1.0)

    rows = np.where(h_proj >= h_thresh)[0]
    cols = np.where(v_proj >= v_thresh)[0]

    if len(rows) == 0 or len(cols) == 0:
        return None  # no content detected

    y_min, y_max = int(rows[0]), int(rows[-1])
    x_min, x_max = int(cols[0]), int(cols[-1])

    pad_y = int(image.height * padding_percent)
    pad_x = int(image.width * padding_percent)

    x = max(0, x_min - pad_x)
    y = max(0, y_min - pad_y)
    x2 = min(image.width, x_max + pad_x)
    y2 = min(image.height, y_max + pad_y)
    w = x2 - x
    h = y2 - y

    # Skip if the crop covers >90% of the image — not a meaningful crop.
    if w * h > image.width * image.height * 0.90:
        return None

    return (x, y, w, h)


# ---------------------------------------------------------------------------
# OCR engine
# ---------------------------------------------------------------------------


class OcrEngine:
    """Wraps TrOCR for inference, with optional per-user LoRA adapters."""

    def __init__(self, model_dir: str | None = None, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model_dir = Path(model_dir or settings.MODEL_DIR)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Loading TrOCR base model (%s) ...", BASE_MODEL_NAME)
        self.processor = TrOCRProcessor.from_pretrained(BASE_MODEL_NAME)
        self.base_model = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL_NAME)
        self.base_model.to(self.device)
        self.base_model.eval()

        # Active model may differ from base_model when a LoRA adapter is loaded.
        self.model: VisionEncoderDecoderModel = self.base_model
        self._active_user_id: Optional[int] = None

        logger.info("TrOCR base model loaded on %s", self.device)

    # ----- LoRA adapter management -----------------------------------------

    def load_user_model(self, user_id: int, lora_path: str) -> None:
        """Load a user's LoRA fine-tuned adapter on top of the base model."""
        from peft import PeftModel

        lora_dir = Path(lora_path)
        if not lora_dir.exists():
            raise FileNotFoundError(f"LoRA weights not found at {lora_dir}")

        logger.info("Loading LoRA adapter for user %d from %s", user_id, lora_dir)

        # Always start from the base model to avoid stacking adapters.
        self.unload_user_model()

        peft_model = PeftModel.from_pretrained(self.base_model, str(lora_dir))
        peft_model.to(self.device)
        peft_model.eval()

        self.model = peft_model
        self._active_user_id = user_id
        logger.info("LoRA adapter for user %d active", user_id)

    def unload_user_model(self) -> None:
        """Revert to the base model (discard any loaded LoRA adapter)."""
        if self._active_user_id is not None:
            logger.info("Unloading LoRA adapter for user %d", self._active_user_id)
        self.model = self.base_model
        self._active_user_id = None

    # ----- Core inference --------------------------------------------------

    def process_single(self, image: Image.Image) -> tuple[str, float]:
        """Run OCR on a single cropped image.

        Returns ``(recognised_text, confidence)`` where confidence is in
        ``[0, 1]``.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=256,
            )

        generated_ids = outputs.sequences
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        confidence = self._compute_confidence(outputs)
        return text, confidence

    def get_confidence(self, image: Image.Image, text: str) -> float:
        """Compute a confidence score for *text* being the correct
        transcription of *image*.

        Uses the model's own generation with ``output_scores=True`` and
        takes the average token log-probability, mapped to ``[0, 1]``.
        """
        _, confidence = self.process_single(image)
        return confidence

    def process_page(
        self,
        image_path: str | Path,
        rotation: int = 0,
        crop: dict | None = None,
        visual_mode: bool = False,
    ) -> list[OcrSegment]:
        """Full pipeline: load image, segment into lines, run OCR.

        Parameters
        ----------
        crop : dict | None
            Optional ``{"x": int, "y": int, "w": int, "h": int}`` crop region
            applied *after* rotation.  Returned bbox coords are offset so they
            align with the full (rotated) image.

        Returns a list of :class:`OcrSegment` instances, one per detected
        text line.
        """
        image = preprocess_image(image_path, rotation=rotation)

        # Apply crop region if provided.
        crop_offset_x = 0
        crop_offset_y = 0
        if crop:
            cx, cy, cw, ch = crop["x"], crop["y"], crop["w"], crop["h"]
            image = image.crop((cx, cy, cx + cw, cy + ch))
            crop_offset_x = cx
            crop_offset_y = cy

        lines = segment_lines(image)

        segments: list[OcrSegment] = []
        for line_img, bbox in lines:
            text, confidence = self.process_single(line_img)
            if text:  # skip empty detections
                # Offset bbox coords by crop origin so overlays align with full image.
                bx, by, bw, bh = bbox
                segments.append(OcrSegment(
                    text=text,
                    confidence=confidence,
                    bbox=(bx + crop_offset_x, by + crop_offset_y, bw, bh),
                ))

        return segments

    # ----- Internal helpers ------------------------------------------------

    @staticmethod
    def _compute_confidence(generate_output) -> float:
        """Derive a ``[0, 1]`` confidence score from generation output.

        ``generate_output`` is the dict returned by ``model.generate`` when
        ``output_scores=True, return_dict_in_generate=True``.  We compute
        the average token-level probability from the logits/scores.
        """
        scores = generate_output.scores  # tuple of (vocab_size,) tensors per step
        if not scores:
            return 0.0

        sequences = generate_output.sequences  # (batch, seq_len)
        # generated_ids includes the decoder start token; scores start from
        # the first *generated* token onward.
        generated_token_ids = sequences[0, 1:]  # drop decoder start token

        log_probs: list[float] = []
        for step_idx, score_tensor in enumerate(scores):
            if step_idx >= len(generated_token_ids):
                break
            token_id = generated_token_ids[step_idx].item()
            # Convert logits to log-probabilities.
            log_softmax = torch.nn.functional.log_softmax(score_tensor[0], dim=-1)
            log_probs.append(log_softmax[token_id].item())

        if not log_probs:
            return 0.0

        avg_log_prob = sum(log_probs) / len(log_probs)
        # Map to [0, 1]:  exp(log_prob) is in (0, 1]; clamp for safety.
        confidence = math.exp(avg_log_prob)
        return max(0.0, min(1.0, confidence))


# ---------------------------------------------------------------------------
# Singleton engine management
# ---------------------------------------------------------------------------

_engine: Optional[OcrEngine] = None
_engine_lock = threading.Lock()


def get_engine() -> OcrEngine:
    """Return the singleton :class:`OcrEngine`, creating it lazily on first call."""
    global _engine
    if not should_try_trocr_fallback():
        raise RuntimeError(
            "TrOCR fallback is disabled because the model is not cached locally "
            "or the feature is turned off in settings."
        )
    if _engine is None:
        with _engine_lock:
            # Double-checked locking.
            if _engine is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                _engine = OcrEngine(model_dir=settings.MODEL_DIR, device=device)
    return _engine


# ---------------------------------------------------------------------------
# Gemini Flash OCR engine (primary, high-quality)
# ---------------------------------------------------------------------------

GEMINI_ROTATION_PROMPT = """Look at this image of handwritten text. The text might be rotated.

What clockwise rotation in degrees (0, 90, 180, or 270) would make the text read normally (upright, left-to-right)?

Reply with ONLY a single number: 0, 90, 180, or 270"""

GEMINI_FAST_OCR_PROMPT = """Analyze this handwritten page for OCR.

Return EVERY handwritten text line and every meaningful hand-drawn arrow.

For text:
{"type":"text","text":"...","box":[y1,x1,y2,x2],"confidence":0.0}

For arrows:
{"type":"visual","element_type":"arrow","label":"optional description","box":[y1,x1,y2,x2],"geometry":{"points":[[x,y],...]}}

Use arrow for ->, =>, curved arrows, or arrows connecting concepts. Only return arrows that are actually visible. Do not invent arrows from text or layout.

Coordinates are normalized 0-1000. For geometry.points use [x,y] pairs in that same normalized coordinate system. For arrows, provide an ordered polyline from tail to arrow tip and preserve the actual direction and bends.

Output ONLY a JSON array. If there is no content, return []."""

GEMINI_DETAILED_OCR_PROMPT = """Analyze this handwritten page as both an OCR system and a document-layout understanding system.

Return EVERY handwritten text line AND every meaningful non-text visual structure.

For text:
{"type":"text","text":"...","box":[y1,x1,y2,x2],"confidence":0.0}

For visual structures:
{"type":"visual","element_type":"arrow|bracket|table|box|circle|underline|connector|diagram","label":"optional description","box":[y1,x1,y2,x2],"geometry":{"points":[[x,y],...],"direction":"optional","rows":0,"columns":0}}

Use arrow for hand-drawn arrows such as ->, =>, curved arrows, or arrows connecting concepts.
Use bracket for (), [], {}, or large hand-drawn brackets that structure content.
Use table for visible tables or rough hand-drawn grids.
Use box for hand-drawn rectangles around text.
Use circle for circled words or numbers.
Use underline for meaningful underlines.
Use connector for lines connecting two regions when they are not arrows.
Use diagram for larger visual structures containing multiple connected shapes/elements.

Only return structures that are actually visible. Do NOT invent structures from text alone.
A visual element may overlap text.

Coordinates are normalized 0-1000. For geometry.points use [x,y] pairs in the same normalized coordinate system.

GEOMETRY IS CRITICAL. The frontend will draw the detected structure from geometry, not from the axis-aligned box:
- arrow: provide an ordered polyline from tail to arrow tip; include at least 2 points and preserve the actual direction and bends.
- connector: provide an ordered polyline along the visible connector.
- underline: provide the actual two endpoints of the underline.
- bracket: provide an ordered polyline following the visible bracket.
- box: provide the visible outline corners as 4 ordered points, even if the box is slightly skewed or hand-drawn.
- circle: provide 4 or more points around the visible outline, plus a center/radius when possible.
- table: provide the outer outline points and accurate rows/columns.
- diagram: provide meaningful visible connector/outline points when possible.
Never replace a diagonal/curved structure with a horizontal line. Never force a hand-drawn outline into a perfect rectangle when its corners are visibly skewed.

Output ONLY a JSON array. If there is no content, return [].""";

GEMINI_PAGE_CORNERS_PROMPT = """This is a camera photo. Is there a paper page, notebook, or open notebook spread visible with non-paper background (desk, table, hands, objects) around it?

If YES — return the 4 corners of ALL the paper/pages visible (the entire notebook spread if open) as JSON:
{"top_left": [x, y], "top_right": [x, y], "bottom_right": [x, y], "bottom_left": [x, y]}
Coordinates are normalized 0-1000 (0=top/left, 1000=bottom/right).

IMPORTANT: Include ALL pages. For an open notebook, give the outer corners of the entire spread (both pages together). Place corners at the OUTER edge of the paper — do NOT trim into any text or page content. It is much better to include a little background than to cut off any text.

If NO — the paper fills nearly all of the image, or no clear paper boundary exists — return exactly: null

Only return corners if there is significant non-paper background visible (more than a thin border). A notebook photo that fills 90%+ of the frame should return null.
Output ONLY valid JSON (the object or null), nothing else."""


def perspective_warp_page(
    image_path: str, corners: list[tuple[int, int]],
) -> str | None:
    """Apply a perspective warp to extract the page quadrilateral as a rectangle.

    Parameters
    ----------
    image_path : str
        Path to the image file on disk.
    corners : list[tuple[int, int]]
        Four corner points [top_left, top_right, bottom_right, bottom_left]
        in pixel coordinates.

    Returns the new image path on success, or None if the warp would produce
    a too-small image (< 100px in either dimension).
    """
    import cv2

    image_path = os.path.normpath(str(image_path).replace("\\", os.sep))
    img = cv2.imread(image_path)
    if img is None:
        logger.warning("perspective_warp_page: could not read %s", image_path)
        return None

    tl, tr, br, bl = corners

    # Compute output dimensions from the max width/height of the quadrilateral.
    width_top = math.hypot(tr[0] - tl[0], tr[1] - tl[1])
    width_bot = math.hypot(br[0] - bl[0], br[1] - bl[1])
    out_w = int(max(width_top, width_bot))

    height_left = math.hypot(bl[0] - tl[0], bl[1] - tl[1])
    height_right = math.hypot(br[0] - tr[0], br[1] - tr[1])
    out_h = int(max(height_left, height_right))

    if out_w < 100 or out_h < 100:
        logger.warning(
            "perspective_warp_page: output too small (%dx%d), skipping",
            out_w, out_h,
        )
        return None

    src_pts = np.float32([tl, tr, br, bl])
    dst_pts = np.float32([
        [0, 0], [out_w - 1, 0],
        [out_w - 1, out_h - 1], [0, out_h - 1],
    ])

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, matrix, (out_w, out_h))

    # Save to a new UUID filename (busts browser cache).
    directory = os.path.dirname(image_path)
    ext = os.path.splitext(image_path)[1] or ".jpg"
    new_path = os.path.join(directory, f"{uuid.uuid4().hex}{ext}")
    cv2.imwrite(new_path, warped)

    # Remove old file.
    try:
        os.remove(image_path)
    except OSError:
        pass

    logger.info(
        "perspective_warp_page: %s → %s (%dx%d)",
        image_path, new_path, out_w, out_h,
    )
    return new_path


def deskew_page(image_path: str) -> str | None:
    """Detect and correct small text/page skew using Hough line detection.

    Uses probabilistic Hough transform to find near-horizontal lines
    (text baselines, ruled notebook lines) and computes the median angle.
    Corrects skew up to ±10°; skips if < 0.5° (already straight enough).

    Returns the new image path, or None if no correction was needed.
    """
    import cv2

    image_path = os.path.normpath(str(image_path).replace("\\", os.sep))
    img = cv2.imread(image_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection for Hough lines.
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines — long minimum length to catch ruled lines/text baselines.
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=100,
        minLineLength=w // 8,
        maxLineGap=20,
    )

    if lines is None:
        logger.info("deskew_page: no Hough lines found, skipping")
        return None

    # Collect angles of near-horizontal lines (within ±15° of horizontal).
    # OpenCV can return several equivalent shapes depending on version and
    # backend (`(N,1,4)`, `(N,4)`, or a flat list), so normalize before
    # unpacking coordinates.
    angles: list[float] = []
    for line in lines:
        arr = np.asarray(line)
        if arr.size == 0:
            continue
        if arr.ndim == 3:
            arr = arr.reshape(-1, 4)
        elif arr.ndim == 2 and arr.shape[1] >= 4:
            arr = arr.reshape(-1, 4)
        elif arr.ndim == 1 and arr.size >= 4:
            arr = arr.reshape(1, 4)
        else:
            continue

        pts = arr[0][:4]
        if len(pts) < 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in pts[:4]]
        a = math.atan2(y2 - y1, x2 - x1) * 180.0 / np.pi
        if abs(a) < 15:
            angles.append(a)

    if len(angles) < 5:
        logger.info("deskew_page: only %d near-horizontal lines, skipping", len(angles))
        return None

    # Use median angle as the skew estimate (robust to outliers).
    angle = float(np.median(angles))

    if abs(angle) < 0.3:
        logger.info("deskew_page: skew %.2f° too small, skipping", angle)
        return None
    if abs(angle) > 10:
        logger.warning("deskew_page: skew %.2f° too large, skipping", angle)
        return None

    # Rotate by -angle to correct the skew.
    # cv2.getRotationMatrix2D positive angle = counter-clockwise.
    correction = -angle
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, correction, 1.0)

    # Expand canvas so rotated image isn't clipped.
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    rotated = cv2.warpAffine(
        img, M, (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # Save to new UUID filename.
    directory = os.path.dirname(image_path)
    ext = os.path.splitext(image_path)[1] or ".jpg"
    new_path = os.path.join(directory, f"{uuid.uuid4().hex}{ext}")
    cv2.imwrite(new_path, rotated)

    try:
        os.remove(image_path)
    except OSError:
        pass

    logger.info("deskew_page: corrected %.2f° — %s → %s", angle, image_path, new_path)
    return new_path


GEMINI_SINGLE_PROMPT = """You are an expert handwriting OCR system. This image shows a cropped region of handwritten text.

Transcribe ALL the handwritten text in this image precisely.
Capture every word, punctuation mark, and number.
Return ONLY the transcribed text, nothing else.
If no text is visible, respond with EMPTY."""


@dataclass
class GeminiOcrResult:
    """Result from Gemini OCR on a full page."""
    rotation: int
    segments: list[OcrSegment]
    visual_elements: list[VisualElement]


OPENAI_OCR_PROMPT = """Read this handwritten page carefully.
Return only the text that is visually supported.
Do not guess unreadable words.
Preserve crossed-out text separately.
Mark unclear regions as [UNCERTAIN].
Detect the language and script.
Return plain text lines in reading order.
"""


class OpenAIOcrEngine:
    """OCR engine using the OpenAI Responses API for handwritten image reading."""

    def __init__(self) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model_name = settings.OPENAI_MODEL or "gpt-5.6"
        logger.info("OpenAI OCR engine initialized (%s)", self.model_name)

    def _call(self, prompt: str, image: Image.Image) -> str:
        import base64

        buffer = __import__("io").BytesIO()
        image.save(buffer, format="JPEG", quality=92)
        image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

        response = self.client.responses.create(
            model=self.model_name,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_data}"},
                    ],
                }
            ],
        )
        text = getattr(response, "output_text", "")
        if text is None:
            text = ""
        return text.strip()

    def process_page(self, image_path: str | Path, rotation: int = 0, crop: dict | None = None) -> "GeminiOcrResult":
        """Read a page image with OpenAI vision and return OCR segments."""
        image = preprocess_image(image_path, rotation=rotation)

        if crop:
            cx, cy, cw, ch = crop["x"], crop["y"], crop["w"], crop["h"]
            image = image.crop((cx, cy, cx + cw, cy + ch))

        try:
            raw_text = self._call(OPENAI_OCR_PROMPT, image)
        except Exception:
            logger.exception("OpenAI OCR API call failed for %s", image_path)
            return GeminiOcrResult(rotation=0, segments=[], visual_elements=[])

        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        if not lines:
            return GeminiOcrResult(rotation=0, segments=[], visual_elements=[])

        width, height = image.size
        spacing = max(int(height / max(len(lines), 1)), 20)
        segments: list[OcrSegment] = []
        for idx, line in enumerate(lines):
            segments.append(
                OcrSegment(
                    text=line,
                    confidence=0.88,
                    bbox=(0, idx * spacing, width, spacing),
                )
            )
        return GeminiOcrResult(rotation=0, segments=segments, visual_elements=[])


class GeminiOcrEngine:
    """OCR engine using Google Gemini Flash multimodal API (new google-genai SDK)."""

    def __init__(self) -> None:
        from google import genai
        from google.genai import types

        # Keep OCR requests bounded. A hung Gemini request must not leave a
        # page stuck in "processing" forever.
        self.client = genai.Client(
            api_key=settings.GEMINI_API_KEY,
            http_options=types.HttpOptions(timeout=30000),
        )
        # Use a model currently available to new Gemini API keys; 2.5 flash
        # variants are no longer available for new users.
        self.model_name = "gemini-3.5-flash-lite"
        logger.info("Gemini OCR engine initialized (%s)", self.model_name)

    def _call(self, prompt: str, image: Image.Image, max_tokens: int = 8192, temperature: float = 0.0) -> str:
        """Send a prompt + image to Gemini and return the text response."""
        import time as _time
        from google.genai import types

        last_exc = None
        for attempt in range(2):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[prompt, image],
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                        response_mime_type="application/json",
                    ),
                )
                text = response.text
                if text is None:
                    logger.warning("Gemini returned None text, finish_reason=%s",
                                   response.candidates[0].finish_reason if response.candidates else "N/A")
                    return ""
                return text.strip()
            except Exception as e:
                last_exc = e
                logger.warning("Gemini API attempt %d/2 failed: %s", attempt + 1, e)
                if attempt < 1:
                    _time.sleep(1)
        raise last_exc

    def detect_rotation(self, image: Image.Image) -> int:
        """Ask Gemini what rotation is needed to make text upright.

        Uses thinking disabled + temperature 0 for consistent results.
        """
        from google.genai import types

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[GEMINI_ROTATION_PROMPT, image],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=16,
                ),
            )
            raw = (response.text or "").strip()
            for token in raw.split():
                token = token.strip("°.,")
                if token in ("0", "90", "180", "270"):
                    return int(token)
            logger.warning("Could not parse rotation from Gemini: %r", raw)
            return 0
        except Exception:
            logger.exception("Gemini rotation detection failed")
            return 0

    def detect_page_corners(
        self, image: Image.Image,
    ) -> list[tuple[int, int]] | None:
        """Ask Gemini to detect the 4 corners of the paper page.

        Returns a list of 4 (x, y) pixel-coordinate tuples
        [top_left, top_right, bottom_right, bottom_left], or None if
        corners can't be detected or the image is already just the page.
        """
        import json as _json
        from google.genai import types

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[GEMINI_PAGE_CORNERS_PROMPT, image],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=256,
                    response_mime_type="application/json",
                ),
            )
            raw = (response.text or "").strip()
        except Exception:
            logger.exception("Gemini page corner detection failed")
            return None

        logger.info("Gemini page corners raw: %.300s", raw)

        if raw.lower() == "null" or not raw:
            return None

        # Strip markdown fences if present.
        json_str = raw
        if json_str.startswith("```"):
            lines = json_str.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            json_str = "\n".join(lines)

        try:
            data = _json.loads(json_str)
        except _json.JSONDecodeError:
            logger.warning("Could not parse page corners JSON: %r", raw)
            return None

        if data is None:
            return None

        img_w, img_h = image.size
        try:
            corners = []
            for key in ("top_left", "top_right", "bottom_right", "bottom_left"):
                pt = data[key]
                px = int(pt[0] / 1000.0 * img_w)
                py = int(pt[1] / 1000.0 * img_h)
                px = max(0, min(px, img_w - 1))
                py = max(0, min(py, img_h - 1))
                corners.append((px, py))

            # Add padding outward from the center to avoid trimming
            # content at the edges.  ~2% of image dimensions.
            cx = sum(c[0] for c in corners) / 4
            cy = sum(c[1] for c in corners) / 4
            pad_x = int(img_w * 0.02)
            pad_y = int(img_h * 0.02)
            padded = []
            for px, py in corners:
                dx = 1 if px < cx else -1 if px > cx else 0
                dy = 1 if py < cy else -1 if py > cy else 0
                # Push corner away from center (outward).
                npx = max(0, min(img_w - 1, px - dx * pad_x))
                npy = max(0, min(img_h - 1, py - dy * pad_y))
                padded.append((npx, npy))
            corners = padded

            return corners
        except (KeyError, TypeError, IndexError, ValueError):
            logger.warning("Invalid page corners structure: %r", data)
            return None

    @staticmethod
    def _refine_visual_geometry(
        image: Image.Image,
        element_type: str,
        bbox: tuple[int, int, int, int],
        geometry: dict | None,
    ) -> dict:
        """Convert Gemini geometry to pixels and refine simple shapes with OpenCV.

        Gemini is good at recognizing *what* a structure is, but its axis-aligned
        bbox is not sufficient for diagonal arrows, hand-drawn boxes, or circles.
        Store real geometry in image pixels so the frontend can render the actual
        direction/outline instead of inventing a horizontal line or perfect CSS
        rectangle.
        """
        import cv2

        x, y, w, h = bbox
        g = dict(geometry) if isinstance(geometry, dict) else {}
        raw_points = g.get("points") or []
        points: list[list[int]] = []

        if isinstance(raw_points, list):
            for point in raw_points:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    continue
                try:
                    px = int(round(float(point[0]) / 1000.0 * image.width))
                    py = int(round(float(point[1]) / 1000.0 * image.height))
                except (TypeError, ValueError):
                    continue
                points.append([
                    max(0, min(image.width - 1, px)),
                    max(0, min(image.height - 1, py)),
                ])

        # Use image evidence to tighten line-like structures. This is deliberately
        # conservative: only replace Gemini geometry when a strong long edge exists.
        line_types = {"arrow", "connector", "underline"}
        if element_type in line_types:
            roi_x0, roi_y0 = max(0, x), max(0, y)
            roi_x1 = min(image.width, x + w)
            roi_y1 = min(image.height, y + h)
            if roi_x1 > roi_x0 and roi_y1 > roi_y0:
                roi = cv2.cvtColor(
                    np.asarray(image.crop((roi_x0, roi_y0, roi_x1, roi_y1))),
                    cv2.COLOR_RGB2GRAY,
                )
                edges = cv2.Canny(roi, 50, 150, apertureSize=3)
                min_len = max(12, int(max(w, h) * 0.35))
                lines = cv2.HoughLinesP(
                    edges,
                    1,
                    np.pi / 180,
                    threshold=max(10, int(min(60, max(w, h) * 0.12))),
                    minLineLength=min_len,
                    maxLineGap=max(8, int(max(w, h) * 0.08)),
                )
                candidates: list[tuple[float, float, list[list[int]]]] = []
                if lines is not None:
                    for line in lines.reshape(-1, 4):
                        lx1, ly1, lx2, ly2 = [int(v) for v in line]
                        dx, dy = lx2 - lx1, ly2 - ly1
                        length = math.hypot(dx, dy)
                        angle = abs(math.degrees(math.atan2(dy, dx)))
                        if element_type == "underline" and angle > 25 and angle < 155:
                            continue
                        candidates.append((
                            length,
                            angle,
                            [[roi_x0 + lx1, roi_y0 + ly1], [roi_x0 + lx2, roi_y0 + ly2]],
                        ))
                if candidates:
                    candidates.sort(key=lambda item: item[0], reverse=True)
                    best = candidates[0]
                    # Don't replace a detailed curved/angled Gemini path with a
                    # short CV fragment. Use CV only when it is clearly substantial.
                    if best[0] >= max(12, 0.45 * max(w, h)) and (
                        element_type == "underline" or len(points) < 2
                    ):
                        points = best[2]

        # Boxes and tables need geometry from the actual ink, not Gemini's
        # axis-aligned detection box. In particular, a table can be recognized
        # correctly while its Gemini bbox is displaced by a large amount.
        if element_type == "table":
            # Search the full page for a strong four-sided table-like outline.
            # Do NOT use Gemini's bbox as the ROI because that is exactly the
            # failure mode this refinement is meant to correct.
            gray_full = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
            edges_full = cv2.Canny(gray_full, 45, 125, apertureSize=3)
            hough = cv2.HoughLinesP(
                edges_full,
                1,
                np.pi / 360,
                threshold=max(30, int(min(image.width, image.height) * 0.055)),
                minLineLength=max(70, int(min(image.width, image.height) * 0.28)),
                maxLineGap=max(12, int(min(image.width, image.height) * 0.035)),
            )

            horizontal: list[tuple[float, float, tuple[int, int, int, int]]] = []
            vertical: list[tuple[float, float, tuple[int, int, int, int]]] = []
            if hough is not None:
                for raw_line in hough.reshape(-1, 4):
                    lx1, ly1, lx2, ly2 = [int(v) for v in raw_line]
                    dx, dy = lx2 - lx1, ly2 - ly1
                    length = math.hypot(dx, dy)
                    angle = math.degrees(math.atan2(dy, dx))
                    if length < 70:
                        continue
                    if abs(angle) <= 22:
                        horizontal.append((length, angle, (lx1, ly1, lx2, ly2)))
                    elif 68 <= abs(angle) <= 112:
                        vertical.append((length, angle, (lx1, ly1, lx2, ly2)))

            # Keep only long, page-internal candidates. Page borders are normally
            # near the image edges and are excluded from the table search.
            margin_x = image.width * 0.06
            margin_y = image.height * 0.06
            horizontal = [
                item for item in horizontal
                if min(item[2][0], item[2][2]) > margin_x
                and max(item[2][0], item[2][2]) < image.width - margin_x
                and min(item[2][1], item[2][3]) > margin_y
                and max(item[2][1], item[2][3]) < image.height - margin_y
            ]
            vertical = [
                item for item in vertical
                if min(item[2][0], item[2][2]) > margin_x
                and max(item[2][0], item[2][2]) < image.width - margin_x
                and min(item[2][1], item[2][3]) > margin_y
                and max(item[2][1], item[2][3]) < image.height - margin_y
            ]

            def line_y(line: tuple[int, int, int, int]) -> float:
                return (line[1] + line[3]) / 2.0

            def line_x(line: tuple[int, int, int, int]) -> float:
                return (line[0] + line[2]) / 2.0

            # Generate plausible outer rectangles from long horizontal and
            # vertical/near-vertical strokes. Score by size, closure, and
            # proximity to Gemini's *center* only. The center is useful as a
            # weak hint, but the Gemini bbox itself is deliberately ignored.
            gem_cx = x + w / 2.0
            gem_cy = y + h / 2.0
            best = None
            best_score = -1.0

            def intersect_lines(a, b):
                ax1, ay1, ax2, ay2 = a
                bx1, by1, bx2, by2 = b
                den = (ax1 - ax2) * (by1 - by2) - (ay1 - ay2) * (bx1 - bx2)
                if abs(den) < 1e-6:
                    return None
                px = ((ax1 * ay2 - ay1 * ax2) * (bx1 - bx2)
                      - (ax1 - ax2) * (bx1 * by2 - by1 * bx2)) / den
                py = ((ax1 * ay2 - ay1 * ax2) * (by1 - by2)
                      - (ay1 - ay2) * (bx1 * by2 - by1 * bx2)) / den
                return px, py

            # Tables in handwritten pages are commonly much wider than tall.
            # Limit candidates to a practical document-table aspect range.
            for top in horizontal:
                for bottom in horizontal:
                    if bottom is top:
                        continue
                    top_line, bottom_line = top[2], bottom[2]
                    if line_y(bottom_line) <= line_y(top_line) + image.height * 0.10:
                        continue
                    left_y = line_y(top_line)
                    right_y = line_y(top_line)
                    for left in vertical:
                        for right in vertical:
                            if line_x(right[2]) <= line_x(left[2]) + image.width * 0.20:
                                continue
                            tl = intersect_lines(top_line, left[2])
                            tr = intersect_lines(top_line, right[2])
                            bl = intersect_lines(bottom_line, left[2])
                            br = intersect_lines(bottom_line, right[2])
                            if not all((tl, tr, bl, br)):
                                continue
                            pts = [tl, tr, br, bl]
                            if any(
                                px < margin_x or px > image.width - margin_x
                                or py < margin_y or py > image.height - margin_y
                                for px, py in pts
                            ):
                                continue

                            width_top = math.dist(tl, tr)
                            width_bottom = math.dist(bl, br)
                            height_left = math.dist(tl, bl)
                            height_right = math.dist(tr, br)
                            width_avg = (width_top + width_bottom) / 2.0
                            height_avg = (height_left + height_right) / 2.0
                            if height_avg <= 0 or width_avg / height_avg < 1.1:
                                continue
                            if width_avg < image.width * 0.35 or height_avg < image.height * 0.15:
                                continue

                            center_x = sum(p[0] for p in pts) / 4.0
                            center_y = sum(p[1] for p in pts) / 4.0
                            center_dist = math.hypot(
                                (center_x - gem_cx) / image.width,
                                (center_y - gem_cy) / image.height,
                            )
                            # Prefer long outlines and reject candidates that are
                            # absurdly far from the recognized table center.
                            if center_dist > 0.55:
                                continue

                            score = (
                                min(2.0, width_avg / image.width)
                                + min(1.2, height_avg / image.height)
                                - center_dist * 1.8
                                + min(0.8, (width_avg / max(height_avg, 1)) * 0.12)
                            )
                            if score > best_score:
                                best_score = score
                                best = pts

            if best is not None:
                points = [
                    [int(round(px)), int(round(py))]
                    for px, py in best
                ]

                # Find only strong internal divider strokes that actually exist.
                # The old frontend invented every row/column, which visibly
                # produced lines that were not present in the handwriting.
                left_x = min(p[0] for p in points)
                right_x = max(p[0] for p in points)
                top_y = min(p[1] for p in points)
                bottom_y = max(p[1] for p in points)
                roi = gray_full[
                    max(0, top_y):min(image.height, bottom_y + 1),
                    max(0, left_x):min(image.width, right_x + 1),
                ]
                if roi.size:
                    inner_edges = cv2.Canny(roi, 45, 125, apertureSize=3)
                    inner_lines = cv2.HoughLinesP(
                        inner_edges, 1, np.pi / 360,
                        threshold=max(18, int(min(roi.shape) * 0.08)),
                        minLineLength=max(45, int(min(roi.shape) * 0.30)),
                        maxLineGap=max(10, int(min(roi.shape) * 0.04)),
                    )
                    grid_lines: list[list[list[int]]] = []
                    if inner_lines is not None:
                        for raw in inner_lines.reshape(-1, 4):
                            qx1, qy1, qx2, qy2 = [int(v) for v in raw]
                            qx1 += left_x; qx2 += left_x
                            qy1 += top_y; qy2 += top_y
                            length = math.hypot(qx2 - qx1, qy2 - qy1)
                            angle = abs(math.degrees(math.atan2(qy2 - qy1, qx2 - qx1)))
                            if length < 0.30 * width_avg:
                                continue
                            if angle <= 20 or 70 <= angle <= 110:
                                # Keep dividers away from the outer perimeter.
                                if (
                                    left_x + 0.08 * width_avg < min(qx1, qx2)
                                    and max(qx1, qx2) < right_x - 0.08 * width_avg
                                ):
                                    grid_lines.append([[qx1, qy1], [qx2, qy2]])

                    g["grid_lines"] = grid_lines
                    # Do not invent rows/columns. They are retained only as
                    # descriptive metadata for the UI if Gemini supplied them.
                    g["rows"] = max(1, int(g.get("rows") or 1))
                    g["columns"] = max(1, int(g.get("columns") or 1))
        # For circles, preserve Gemini's ellipse geometry when supplied. If not,
        # use the bbox as an ellipse instead of a rounded CSS rectangle.
        if element_type == "circle":
            if len(points) < 4:
                g["center"] = [x + w / 2, y + h / 2]
                g["radius"] = [w / 2, h / 2]
            else:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                g["center"] = [sum(xs) / len(xs), sum(ys) / len(ys)]
                g["radius"] = [max(1, (max(xs) - min(xs)) / 2), max(1, (max(ys) - min(ys)) / 2)]

        if points:
            g["points"] = points

        if element_type == "arrow" and len(points) >= 2:
            g["direction"] = [
                points[-1][0] - points[-2][0],
                points[-1][1] - points[-2][1],
            ]

        if element_type == "table":
            g["rows"] = max(1, int(g.get("rows") or 1))
            g["columns"] = max(1, int(g.get("columns") or 1))

        # Persist an explicit coordinate-space marker so older database rows
        # containing Gemini's normalized 0-1000 geometry remain distinguishable.
        g["coordinate_space"] = "pixel"

        return g

    def process_page(
        self,
        image_path: str | Path,
        rotation: int = 0,
        crop: dict | None = None,
        visual_mode: bool = False,
    ) -> GeminiOcrResult:
        """Send full page image to Gemini for OCR.

        Uses Gemini for text recognition (high accuracy) and CV-based
        line detection for bounding boxes (pixel-precise alignment).

        Strategy: Gemini provides accurate text but unreliable bounding
        boxes (tends to return a uniform grid). So we use Gemini text
        paired with CV-detected line positions (with ruled-line removal
        to handle notebook paper).
        """

        image = preprocess_image(image_path, rotation=rotation)

        crop_offset_x = 0
        crop_offset_y = 0
        if crop:
            cx, cy, cw, ch = crop["x"], crop["y"], crop["w"], crop["h"]
            image = image.crop((cx, cy, cx + cw, cy + ch))
            crop_offset_x = cx
            crop_offset_y = cy

        img_width, img_height = image.size

        # ── Step 1: Get text from Gemini ─────────────────────────────
        try:
            raw_text = self._call(
                GEMINI_DETAILED_OCR_PROMPT if visual_mode else GEMINI_FAST_OCR_PROMPT,
                image,
                temperature=0.0,
            )
            logger.info(
                "Gemini OCR raw response (%d chars): %.300s",
                len(raw_text), raw_text,
            )
        except Exception:
            logger.exception("Gemini OCR API call failed for %s", image_path)
            return GeminiOcrResult(rotation=0, segments=[], visual_elements=[])

        if not raw_text or raw_text.strip() == "[]":
            return GeminiOcrResult(rotation=0, segments=[], visual_elements=[])

        # Extract text lines and visual structures from Gemini response.
        entries = self._parse_gemini_json(raw_text)
        visual_elements: list[VisualElement] = []
        if entries is not None:
            clean_entries: list[dict] = []
            text_lines: list[str] = []
            import json as _json
            for e in entries:
                if not isinstance(e, dict):
                    continue
                item_type = str(e.get("type") or "text").strip().lower()
                box = e.get("box") or e.get("box_2d")
                if item_type == "visual":
                    if not isinstance(box, list) or len(box) != 4:
                        continue
                    try:
                        y1, x1, y2, x2 = [float(v) for v in box]
                        px1 = max(0, int(x1 / 1000.0 * img_width))
                        py1 = max(0, int(y1 / 1000.0 * img_height))
                        px2 = min(img_width, int(x2 / 1000.0 * img_width))
                        py2 = min(img_height, int(y2 / 1000.0 * img_height))
                        element_type = str(e.get("element_type") or "diagram").lower()
                        visual_bbox = (px1, py1, max(1, px2 - px1), max(1, py2 - py1))
                        refined_geometry = self._refine_visual_geometry(
                            image,
                            element_type,
                            visual_bbox,
                            e.get("geometry"),
                        )
                        visual_elements.append(
                            VisualElement(
                                element_type=element_type,
                                confidence=max(0.0, min(1.0, float(e.get("confidence", 0.9)))),
                                bbox=visual_bbox,
                                label=e.get("label"),
                                geometry=_json.dumps(refined_geometry, ensure_ascii=False),
                            )
                        )
                    except (TypeError, ValueError):
                        logger.warning("Skipping malformed visual element: %r", e)
                    continue
                t = (e.get("text") or e.get("text_content") or "").strip()
                if not t:
                    continue
                clean_entries.append({"text": t, "box": box})
                text_lines.append(t)
            entries = clean_entries
        else:
            # Plain text fallback — skip if it looks like broken JSON.
            stripped = raw_text.strip()
            if (stripped.startswith("[") or stripped.startswith("{")
                    or stripped.startswith("```")):
                logger.warning("Gemini returned unparseable JSON, returning empty")
                return GeminiOcrResult(rotation=0, segments=[], visual_elements=[])
            text_lines = [l.strip() for l in raw_text.split("\n") if l.strip()]

        if not text_lines:
            return GeminiOcrResult(rotation=0, segments=[], visual_elements=[])

        # ── Step 2: Use Gemini's bounding boxes with spacing correction ─
        # Gemini's x-coordinates are accurate but y-spacing drifts on
        # long pages (uniform grid vs actual ruled-line spacing).
        # Detect actual line spacing from the image and re-space lines.
        segments = self._build_direct_segments(
            text_lines, entries, image, img_width, img_height,
            crop_offset_x, crop_offset_y,
        )

        logger.info(
            "Gemini OCR: %d text lines → %d segments for %s",
            len(text_lines), len(segments), image_path,
        )
        return GeminiOcrResult(rotation=0, segments=segments, visual_elements=visual_elements)

    def _build_direct_segments(
        self,
        text_lines: list[str],
        entries: list | None,
        image: Image.Image,
        img_width: int,
        img_height: int,
        crop_offset_x: int = 0,
        crop_offset_y: int = 0,
    ) -> list[OcrSegment]:
        """Build segments using Gemini's bounding boxes with spacing correction.

        Gemini's x-coordinates are accurate, but y-coordinates use a uniform
        grid that drifts from actual ruled-line spacing on notebook pages.
        We detect the real line spacing from the image via autocorrelation
        and re-space lines accordingly.
        """
        n = len(text_lines)
        if n == 0:
            return []

        # ── Parse Gemini boxes into pixel coords ──────────────────
        raw_boxes: list[tuple[int, int, int, int] | None] = []
        for i in range(n):
            box = None
            if entries and i < len(entries) and isinstance(entries[i], dict):
                box = entries[i].get("box") or entries[i].get("box_2d")
            if isinstance(box, list) and len(box) == 4:
                y1, x1, y2, x2 = box
                py1 = max(0, int(y1 / 1000.0 * img_height))
                px1 = max(0, int(x1 / 1000.0 * img_width))
                py2 = min(img_height, int(y2 / 1000.0 * img_height))
                px2 = min(img_width, int(x2 / 1000.0 * img_width))
                raw_boxes.append((px1, py1, px2, py2))
            else:
                raw_boxes.append(None)

        # ── Detect actual ink line positions from the image ─────────
        # Gemini's y-coordinates drift on long pages.  We detect where
        # ink actually sits using binarization + horizontal projection,
        # then snap each Gemini text line to the nearest ink line.
        ink_lines = self._detect_ink_lines(image)

        gemini_ys = []
        for b in raw_boxes:
            gemini_ys.append(b[1] if b else None)

        # Match Gemini text lines to ink lines (both in top-to-bottom
        # order).  Each Gemini line snaps to the nearest unmatched ink
        # line; if no ink line is close, keep Gemini's y.
        corrected_ys: list[int | None] = list(gemini_ys)
        corrected_heights: list[int | None] = [None] * n
        used_ink: set[int] = set()

        if ink_lines:
            # Compute median ink spacing and height for filtering.
            ink_spacings = [ink_lines[k + 1][0] - ink_lines[k][0]
                            for k in range(len(ink_lines) - 1)]
            med_spacing = (sorted(ink_spacings)[len(ink_spacings) // 2]
                           if ink_spacings else 110)
            ink_heights = sorted(il[3] for il in ink_lines)
            med_height = ink_heights[len(ink_heights) // 2]
            max_match_dist = int(med_spacing * 1.5)

            ink_idx = 0
            last_matched_ink = -1
            last_matched_gemini = -1

            # Identify body-text start: find first line with typical
            # spacing to the next line (skip header/date lines).
            gemini_spacings = []
            for i in range(n - 1):
                if gemini_ys[i] is not None and gemini_ys[i + 1] is not None:
                    gemini_spacings.append(gemini_ys[i + 1] - gemini_ys[i])
            med_gemini_spacing = (sorted(gemini_spacings)[len(gemini_spacings) // 2]
                                  if gemini_spacings else 0)

            body_start = 0
            if med_gemini_spacing > 0:
                for i in range(n - 1):
                    if gemini_ys[i] is None or gemini_ys[i + 1] is None:
                        continue
                    gap = gemini_ys[i + 1] - gemini_ys[i]
                    if abs(gap - med_gemini_spacing) < med_gemini_spacing * 0.3:
                        body_start = i
                        break

            for i in range(n):
                gy = gemini_ys[i]
                if gy is None:
                    continue

                # Header lines before the body block keep Gemini positions.
                if i < body_start:
                    continue

                # Sequential matching: take the NEXT available ink line.
                j = ink_idx
                while j < len(ink_lines):
                    if j in used_ink:
                        j += 1
                        continue
                    # Skip oversized blobs (likely merged header area).
                    if ink_lines[j][3] > med_height * 2.5:
                        used_ink.add(j)
                        j += 1
                        continue
                    break

                if j < len(ink_lines):
                    center, top, bot, ht = ink_lines[j]
                    if abs(gy - center) < max_match_dist * 2:
                        corrected_ys[i] = top
                        corrected_heights[i] = ht
                        used_ink.add(j)
                        ink_idx = j + 1
                        last_matched_ink = j
                        last_matched_gemini = i

            # Extrapolate for unmatched lines at the end using
            # spacing from the last few matched lines.
            if last_matched_gemini >= 0 and last_matched_gemini < n - 1:
                last_y = corrected_ys[last_matched_gemini]
                for i in range(last_matched_gemini + 1, n):
                    if corrected_heights[i] is not None:
                        continue  # already matched
                    offset = i - last_matched_gemini
                    corrected_ys[i] = last_y + offset * med_spacing
                    corrected_heights[i] = med_height

            matched = sum(1 for i in range(n)
                          if corrected_heights[i] is not None)
            logger.info(
                "Ink-line matching: %d/%d Gemini lines snapped to ink "
                "(median spacing=%dpx, median height=%dpx)",
                matched, n, med_spacing, med_height,
            )

        # ── Build segments ────────────────────────────────────────
        # Default box height: median ink line height, or Gemini box height.
        default_bh = 80
        if ink_lines:
            heights = sorted(il[3] for il in ink_lines)
            default_bh = heights[len(heights) // 2]

        segments: list[OcrSegment] = []
        for i, text in enumerate(text_lines):
            b = raw_boxes[i]
            if b is not None:
                px1, _, px2, _ = b
                py1 = corrected_ys[i] if corrected_ys[i] is not None else b[1]
                bw = max(px2 - px1, 5)
                bh = corrected_heights[i] if corrected_heights[i] else default_bh
            else:
                spacing = img_height / max(n + 1, 1)
                py1 = int(i * spacing)
                px1, px2 = 0, img_width
                bw = img_width
                bh = int(spacing)

            # Clamp to image bounds.
            py1 = max(0, min(py1, img_height - bh))

            segments.append(OcrSegment(
                text=text,
                confidence=0.90,
                bbox=(px1 + crop_offset_x, py1 + crop_offset_y, bw, bh),
            ))

        return segments

    @staticmethod
    def _detect_ink_lines(
        image: Image.Image,
    ) -> list[tuple[int, int, int, int]]:
        """Detect text line positions from ink in the image.

        Returns list of (center_y, top_y, bottom_y, height) tuples,
        sorted top-to-bottom.  Uses Otsu binarization + horizontal
        projection to find rows with ink.
        """
        import cv2

        gray = np.array(image.convert("L"))
        h, w = gray.shape
        if h < 200 or w < 200:
            return []

        # Binarize (ink = white in result).
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )

        # Use middle 75% of width to avoid margin noise.
        text_area = binary[:, w // 8: 7 * w // 8]

        # Horizontal projection.
        proj = text_area.sum(axis=1).astype(float) / 255

        # Smooth.
        kernel = np.ones(3) / 3
        proj_s = np.convolve(proj, kernel, mode="same")

        # Threshold: rows with significant ink.
        nz = proj_s[proj_s > 0]
        if len(nz) == 0:
            return []
        threshold = max(float(np.percentile(nz, 25)), 20.0)

        # Find contiguous ink runs.
        in_text = False
        line_start = 0
        raw_lines: list[tuple[int, int, int, int]] = []
        for y in range(h):
            if proj_s[y] > threshold:
                if not in_text:
                    line_start = y
                    in_text = True
            else:
                if in_text:
                    ht = y - line_start
                    if ht > 15:
                        center = (line_start + y) // 2
                        raw_lines.append((center, line_start, y, ht))
                    in_text = False

        if not raw_lines:
            return []

        # Split oversized blobs that likely contain multiple lines.
        median_ht = sorted(r[3] for r in raw_lines)[len(raw_lines) // 2]
        split_threshold = median_ht * 2

        result: list[tuple[int, int, int, int]] = []
        for center, top, bot, ht in raw_lines:
            if ht > split_threshold and median_ht > 15:
                # Split into sub-lines using projection valleys.
                sub_proj = proj_s[top:bot]
                min_val = sub_proj.min()
                split_thresh = (sub_proj.max() + min_val) / 3

                # Find valleys (local minima below threshold).
                splits = [0]
                for y in range(10, len(sub_proj) - 10):
                    if (sub_proj[y] < split_thresh
                            and sub_proj[y] <= sub_proj[y - 5]
                            and sub_proj[y] <= sub_proj[y + 5]
                            and (not splits or y - splits[-1] > median_ht * 0.5)):
                        splits.append(y)
                splits.append(len(sub_proj))

                for si in range(len(splits) - 1):
                    s_top = top + splits[si]
                    s_bot = top + splits[si + 1]
                    s_ht = s_bot - s_top
                    if s_ht > 10:
                        result.append(((s_top + s_bot) // 2, s_top, s_bot, s_ht))
            else:
                result.append((center, top, bot, ht))

        result.sort(key=lambda x: x[0])
        logger.info("Detected %d ink lines from image", len(result))
        return result

    def _calibrate_spacing(
        self,
        image: Image.Image,
        text_lines: list[str],
        raw_boxes: list,
        gemini_ys: list,
        anchor_idx: int | None,
        anchor_y: int | None,
        gemini_spacing: int | None,
        img_width: int,
        img_height: int,
    ) -> int | None:
        """Calibrate line spacing by OCR-verifying one line near the page end.

        Searches for the correct y-position of a line at ~75% through the
        page, then computes exact spacing from anchor to that position.
        Uses 1-7 Gemini API calls.
        """
        if anchor_idx is None or anchor_y is None or gemini_spacing is None:
            return None
        if len(text_lines) < 8:
            return None  # too few lines to calibrate

        # Pick a calibration line at ~75% through the body text.
        cal_idx = anchor_idx + max(1, int((len(text_lines) - anchor_idx) * 0.75))
        cal_idx = min(cal_idx, len(text_lines) - 1)
        if cal_idx <= anchor_idx:
            return None

        cal_text = text_lines[cal_idx]
        cal_words = set(cal_text.lower().split()[:5])
        if len(cal_words) < 2:
            return None

        # Get x-bounds from the box (or use defaults).
        b = raw_boxes[cal_idx] if cal_idx < len(raw_boxes) else None
        if b is not None:
            bx1, _, bx2, _ = b
        else:
            bx1, bx2 = 0, img_width

        # Approximate spacing from autocorrelation for initial estimate.
        approx_spacing = self._detect_line_spacing(image) or gemini_spacing
        # Estimate y position.
        est_y = anchor_y + (cal_idx - anchor_idx) * approx_spacing
        box_h = approx_spacing

        # Search ±150px around estimate in 15px steps.
        best_y = None
        best_overlap = 0.0
        for offset in range(0, 160, 15):
            for sign in [0, -1, 1] if offset == 0 else [-1, 1]:
                y = int(est_y + sign * offset)
                if y < 0 or y + box_h > img_height:
                    continue
                crop = image.crop((bx1, y, bx2, y + box_h))
                text, _ = self.process_single(crop)
                got_words = set((text or "").lower().split()[:8])
                overlap = len(cal_words & got_words) / len(cal_words)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_y = y
                if overlap >= 0.5:
                    break
            if best_overlap >= 0.5:
                break

        if best_y is None or best_overlap < 0.4:
            logger.info("Spacing calibration failed for line %d", cal_idx)
            return None

        # Compute actual spacing from anchor to calibration point.
        lines_between = cal_idx - anchor_idx
        actual = int(round((best_y - anchor_y) / lines_between))
        logger.info(
            "Spacing calibrated: line %d at y=%d (est=%d, offset=%+d), "
            "spacing=%dpx (%.1f%% match)",
            cal_idx, best_y, int(est_y), best_y - int(est_y),
            actual, best_overlap * 100,
        )
        return actual

    @staticmethod
    def _detect_line_spacing(image: Image.Image) -> int | None:
        """Detect ruled-line spacing from a notebook page image.

        Uses autocorrelation of the horizontal edge projection to find
        the dominant periodic spacing (the distance between ruled lines).
        """
        import cv2

        gray = np.array(image.convert("L"))
        h, w = gray.shape

        if h < 200 or w < 200:
            return None

        # Use middle 80% of width to avoid margin noise.
        text_area = gray[:, w // 10 : 9 * w // 10]
        edges = cv2.Canny(text_area, 30, 100)
        proj = edges.sum(axis=1).astype(float)

        # Autocorrelation.
        proj_norm = proj - proj.mean()
        norm_sq = float(np.dot(proj_norm, proj_norm))
        if norm_sq < 1e-6:
            return None
        autocorr = np.correlate(proj_norm, proj_norm, mode="full")
        autocorr = autocorr[len(autocorr) // 2:]
        autocorr = autocorr / autocorr[0]

        # Find first significant peak in range [60, 200] pixels.
        best_lag = None
        best_val = 0.0
        for lag in range(60, min(200, len(autocorr) - 1)):
            if (autocorr[lag] > autocorr[lag - 1]
                    and autocorr[lag] > autocorr[lag + 1]
                    and autocorr[lag] > 0.3
                    and autocorr[lag] > best_val):
                best_lag = lag
                best_val = float(autocorr[lag])
                break  # first peak is the fundamental

        if best_lag is not None:
            logger.info(
                "Detected line spacing: %dpx (autocorr=%.3f)",
                best_lag, best_val,
            )
        return best_lag

    @staticmethod
    def _build_cv_aligned_segments(
        text_lines: list[str],
        entries: list | None,
        image: Image.Image,
        img_width: int,
        img_height: int,
        crop_offset_x: int = 0,
        crop_offset_y: int = 0,
    ) -> list[OcrSegment]:
        """Build segments using CV peak detection with ordered matching.

        Gemini returns text in correct reading order but bounding boxes
        can be unreliable (bunched together, overlapping).  We use
        Gemini y-coordinates as approximate targets, enforce minimum
        spacing between entries, then snap each target to the nearest
        CV ink peak while maintaining monotonic order.

        This handles both well-spaced coordinates (page 1) and
        bunched/overlapping coordinates (page 2) correctly.
        """
        import cv2

        n = len(text_lines)
        if n == 0:
            return []

        # ── Extract Gemini per-line coordinates ──────────────────────
        gemini_centers: list[int | None] = []
        per_line_x: list[tuple[int, int]] = []

        if entries is not None:
            for e in entries:
                box = e.get("box") or e.get("box_2d") if isinstance(e, dict) else None
                if isinstance(box, list) and len(box) == 4:
                    y1, x1, y2, x2 = box
                    yt = int(y1 / 1000.0 * img_height)
                    yb = int(y2 / 1000.0 * img_height)
                    gemini_centers.append((yt + yb) // 2)
                    px1 = max(0, int(x1 / 1000.0 * img_width))
                    px2 = min(img_width, int(x2 / 1000.0 * img_width))
                    per_line_x.append((px1, px2))
                else:
                    gemini_centers.append(None)
                    per_line_x.append((0, img_width))

        while len(gemini_centers) < n:
            gemini_centers.append(None)
        while len(per_line_x) < n:
            per_line_x.append((0, img_width))

        # ── Build CV handwriting image (ruled lines removed) ─────────
        gray = np.array(image.convert("L"))
        img_h, img_w = gray.shape

        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
        line_len = max(img_w // 4, 100)
        horiz_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (line_len, 1),
        )
        ruled = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)
        ruled = cv2.dilate(
            ruled,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)),
            iterations=1,
        )
        handwriting = cv2.subtract(binary, ruled)

        expected_spacing = img_height / max(n + 2, 1)

        # ── Per-entry projection cache ──────────────────────────────
        # Cache projections by (x1, x2) so shared x-ranges are fast.
        _proj_cache: dict[tuple[int, int], np.ndarray] = {}

        def _get_proj(col_x1: int, col_x2: int) -> np.ndarray:
            xpad = int(img_w * 0.02)
            cx1 = max(0, col_x1 - xpad)
            cx2 = min(img_w, col_x2 + xpad)
            key = (cx1, cx2)
            if key not in _proj_cache:
                if cx2 <= cx1:
                    _proj_cache[key] = np.zeros(img_h, dtype=np.float32)
                else:
                    p = handwriting[:, cx1:cx2].astype(
                        np.float32,
                    ).sum(axis=1) / 255.0
                    p = cv2.GaussianBlur(
                        p.reshape(-1, 1), (1, 15), 2,
                    ).flatten()
                    _proj_cache[key] = p
            return _proj_cache[key]

        # ── Greedy ordered matching ─────────────────────────────────
        # For each entry in reading order:
        # 1. Use Gemini center as target, enforce min spacing
        # 2. Find nearest CV peak in that entry's x-range
        # 3. Advance search window past the chosen peak

        min_spacing = max(int(expected_spacing * 0.3), 10)
        max_snap_dist = int(expected_spacing * 2)

        assigned_centers: list[int] = []

        for i in range(n):
            gc = gemini_centers[i]
            if gc is None:
                if assigned_centers:
                    gc = assigned_centers[-1] + int(expected_spacing)
                else:
                    gc = int(expected_spacing)

            # Enforce minimum spacing from previous assignment.
            if assigned_centers:
                gc = max(gc, assigned_centers[-1] + min_spacing)

            # Search for nearest peak in this entry's x-range.
            lx1, lx2 = per_line_x[i]
            proj = _get_proj(lx1, lx2)
            thr = max(proj.max() * 0.08, 1.0)

            search_from = max(1, (assigned_centers[-1] + min_spacing) if assigned_centers else 0)
            search_to = min(len(proj) - 1, gc + max_snap_dist)

            best_y = gc
            best_dist = max_snap_dist + 1
            for j in range(search_from, search_to):
                if (proj[j] > thr
                        and proj[j] > proj[j - 1]
                        and proj[j] >= proj[j + 1]):
                    d = abs(j - gc)
                    if d < best_dist:
                        best_dist = d
                        best_y = j

            # Enforce monotonic ordering.
            if assigned_centers and best_y <= assigned_centers[-1]:
                best_y = assigned_centers[-1] + min_spacing

            assigned_centers.append(best_y)

        # ── Compute line height ──────────────────────────────────────
        # Use expected_spacing (image height / entry count) which
        # closely matches the actual line-to-line distance on the page.
        typical_h = max(20, int(expected_spacing))

        logger.info(
            "CV alignment: %d entries, typical_h=%d, "
            "first=%d, last=%d, expected_spacing=%.0f",
            n, typical_h,
            assigned_centers[0] if assigned_centers else -1,
            assigned_centers[-1] if assigned_centers else -1,
            expected_spacing,
        )

        # ── Build segments with per-line ink x-detection ────────────
        # Gemini x-coordinates often include page margins.  For each
        # line, detect the actual ink extent in the handwriting image
        # to get tighter, more accurate x-bounds.
        segments: list[OcrSegment] = []
        for i, text in enumerate(text_lines):
            center = assigned_centers[i]
            bh = typical_h
            by = center - bh // 2
            by = max(0, min(by, img_height - 1))
            bh = max(20, min(bh, img_height - by))

            lx1, lx2 = per_line_x[i]

            # Detect actual ink extent in this line's y-range.
            y_start = max(0, by)
            y_end = min(img_h, by + bh)
            if y_end > y_start:
                # Search within Gemini's x-range ± 10% margin.
                margin = max(int((lx2 - lx1) * 0.1), 20)
                sx1 = max(0, lx1 - margin)
                sx2 = min(img_w, lx2 + margin)
                line_slice = handwriting[y_start:y_end, sx1:sx2]
                col_sums = line_slice.astype(
                    np.float32,
                ).sum(axis=0) / 255.0
                ink_thr = max(col_sums.max() * 0.1, 1.0)
                ink_cols = np.where(col_sums >= ink_thr)[0]

                if len(ink_cols) >= 5:
                    # Use detected ink bounds with small padding.
                    hpad = max(8, img_w // 300)
                    bx = max(0, sx1 + int(ink_cols[0]) - hpad)
                    bx_end = min(
                        img_w, sx1 + int(ink_cols[-1]) + hpad,
                    )
                    bw = bx_end - bx
                else:
                    # Fallback to Gemini x-coords.
                    hpad = max(5, img_w // 200)
                    bx = max(0, lx1 - hpad)
                    bw = min(img_w, lx2 + hpad) - bx
            else:
                hpad = max(5, img_w // 200)
                bx = max(0, lx1 - hpad)
                bw = min(img_w, lx2 + hpad) - bx

            segments.append(OcrSegment(
                text=text,
                confidence=0.95,
                bbox=(
                    bx + crop_offset_x,
                    by + crop_offset_y,
                    bw,
                    bh,
                ),
            ))

        return segments

    @staticmethod
    def _parse_gemini_json(raw_text: str) -> list | None:
        """Parse Gemini JSON response, stripping markdown fences. Returns None on failure."""
        import json as _json
        import re

        json_str = raw_text.strip()
        if json_str.startswith("```"):
            lines = json_str.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            json_str = "\n".join(lines)

        try:
            entries = _json.loads(json_str)
        except _json.JSONDecodeError:
            # Try to extract a JSON array from the response (Gemini sometimes
            # wraps it in extra text or has trailing commas).
            match = re.search(r'\[.*\]', json_str, re.DOTALL)
            if match:
                try:
                    # Remove trailing commas before ] which is invalid JSON
                    cleaned = re.sub(r',\s*([}\]])', r'\1', match.group())
                    entries = _json.loads(cleaned)
                    if isinstance(entries, list):
                        return entries
                except _json.JSONDecodeError:
                    pass
            logger.warning("Failed to parse Gemini JSON response")
            return None

        if not isinstance(entries, list):
            return None
        return entries

    def process_single(self, image: Image.Image) -> tuple[str, float]:
        """Run OCR on a single cropped image region.

        Returns (text, confidence).
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        try:
            text = self._call(GEMINI_SINGLE_PROMPT, image, max_tokens=1024)
            if text == "EMPTY":
                return "", 0.0
            return text, 0.95
        except Exception:
            logger.exception("Gemini single OCR failed")
            return "", 0.0


# Singleton Gemini engine
_gemini_engine: Optional[GeminiOcrEngine] = None
def _text_overlap(a: str, b: str) -> float:
    """Return word-level overlap ratio between two strings (0-1)."""
    if not a or not b:
        return 0.0
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    return len(intersection) / max(len(words_a), len(words_b))


def _text_similar(expected: str, got: str, threshold: float = 0.4) -> bool:
    """Check if OCR output is similar enough to expected text."""
    if not expected or not got:
        return False
    return _text_overlap(expected, got) >= threshold


_gemini_lock = threading.Lock()
_openai_engine: Optional[OpenAIOcrEngine] = None
_openai_lock = threading.Lock()


def has_gemini() -> bool:
    """Return True if a Gemini API key is configured."""
    return bool(settings.GEMINI_API_KEY)


def has_openai() -> bool:
    """Return True if an OpenAI API key is configured."""
    return bool(settings.OPENAI_API_KEY)


def get_gemini_engine() -> GeminiOcrEngine:
    """Return the singleton GeminiOcrEngine, creating it lazily."""
    global _gemini_engine
    if _gemini_engine is None:
        with _gemini_lock:
            if _gemini_engine is None:
                _gemini_engine = GeminiOcrEngine()
    return _gemini_engine


def get_openai_engine() -> OpenAIOcrEngine:
    """Return the singleton OpenAIOcrEngine, creating it lazily."""
    global _openai_engine
    if _openai_engine is None:
        with _openai_lock:
            if _openai_engine is None:
                _openai_engine = OpenAIOcrEngine()
    return _openai_engine
