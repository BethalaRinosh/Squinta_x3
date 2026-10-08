from PIL import Image, ImageDraw

from app.ink_layers import InkRegion, InkLayerAnalyzer


def _make_page_image():
    image = Image.new("RGB", (200, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 20), "hello", fill="black")
    draw.line((10, 50, 190, 50), fill="black", width=3)
    return image


def test_ink_layer_analyzer_returns_safe_fallback_and_detects_revision_like_regions():
    image = _make_page_image()
    regions = InkLayerAnalyzer().analyze_image(image)

    assert regions
    assert any(region.label == "MAIN_INK" for region in regions)
    assert any(region.label in {"CROSSED_OUT_INK", "MARGIN_NOTE", "UNKNOWN"} for region in regions)

    main_regions = [r for r in regions if r.label == "MAIN_INK"]
    assert main_regions
    assert main_regions[0].score >= 0.0

    revision = next((r for r in regions if r.relationship and r.relationship.get("type") == "revision"), None)
    assert revision is not None
    assert revision.relationship["deleted_text"]
