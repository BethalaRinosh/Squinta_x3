from PIL import Image, ImageDraw

from app.ocr import _calculate_gemini_confidence


def test_confidence_varies_with_ocr_evidence():
    clear = Image.new("RGB", (400, 200), "white")
    draw = ImageDraw.Draw(clear)
    draw.text((20, 60), "hello world", fill="black")

    dense = Image.new("RGB", (400, 200), "white")
    draw = ImageDraw.Draw(dense)
    for y in range(20, 180, 12):
        draw.line((5, y, 395, y), fill="black", width=2)

    good = _calculate_gemini_confidence("hello world", clear, 0.95)
    uncertain = _calculate_gemini_confidence("[UNCERTAIN] hello", clear, 0.95)
    noisy = _calculate_gemini_confidence("x", dense, 0.95)

    assert 0.0 <= good <= 1.0
    assert 0.0 <= uncertain < good
    assert 0.0 <= noisy < good


def test_empty_confidence_is_zero():
    image = Image.new("RGB", (100, 100), "white")
    assert _calculate_gemini_confidence("", image, 0.95) == 0.0
