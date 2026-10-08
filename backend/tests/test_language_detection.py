from app.language_detection import (
    translate_text,
    LanguageDetector,
    SUPPORTED_LANGUAGES,
    build_ocr_language_annotation,
    detect_language,
    detect_language_spans,
    identify_language_and_translate_to_english,
)


def test_registry_contains_indian_languages_and_english():
    expected = {
        "as", "bn", "brx", "doi", "gu", "hi", "kn", "ks", "kok", "mai",
        "ml", "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta",
        "te", "ur", "en",
    }

    assert expected.issubset(SUPPORTED_LANGUAGES)


def test_detects_pure_tamil_text():
    result = LanguageDetector().detect_language("நாளைக்கு சந்திப்பு")

    assert result.language == "ta"
    assert result.script == "Tamil"
    assert result.confidence > 0.9
    assert result.spans[0].language_name == "Tamil"


def test_detects_pure_hindi_text_as_devanagari_with_language_uncertainty():
    result = LanguageDetector().detect_language("कल बैठक है")

    assert result.language == "hi"
    assert result.script == "Devanagari"
    assert result.confidence < 0.9
    assert result.alternatives
    assert any(alt.language == "mr" for alt in result.alternatives)


def test_detects_mixed_english_tamil_numbers():
    spans = detect_language_spans("Meeting நாளைக்கு at 10 AM")

    assert [(span.text, span.language, span.script) for span in spans] == [
        ("Meeting", "en", "Latin"),
        ("நாளைக்கு", "ta", "Tamil"),
        ("at", "en", "Latin"),
        ("10", "numeric", "Common"),
        ("AM", "en", "Latin"),
    ]

    result = LanguageDetector().detect_language("Meeting நாளைக்கு at 10 AM")
    assert result.language == "mixed"
    assert any(alt.language == "ta" for alt in result.alternatives)
    assert any(alt.language == "en" for alt in result.alternatives)


def test_detects_malayalam_telugu_kannada_bengali_gujarati():
    detector = LanguageDetector()

    assert detector.detect_language("നാളെ യോഗം").language == "ml"
    assert detector.detect_language("రేపు సమావేశం").language == "te"
    assert detector.detect_language("ನಾಳೆ ಸಭೆ").language == "kn"
    assert detector.detect_language("আগামীকাল সভা").language == "bn"
    assert detector.detect_language("કાલે બેઠક").language == "gu"


def test_ocr_language_annotation_is_exposed_for_results():
    result = detect_language("நாளைக்கு சந்திப்பு")

    assert result.language == "ta"
    assert result.language_name == "Tamil"
    assert result.script == "Tamil"
    assert result.confidence > 0.9


def test_build_ocr_language_annotation_returns_normalized_result():
    data = build_ocr_language_annotation("Meeting நாளைக்கு at 10 AM")

    assert data["language"] == "mixed"
    assert data["language_name"] == "Mixed"
    assert data["script"] == "Mixed"
    assert data["confidence"] > 0


def test_identify_language_and_translate_to_english_uses_detector_and_translator(monkeypatch):
    def fake_translate(text, src, dest):
        assert src == "ta"
        assert dest == "en"
        return "Meet tomorrow"

    monkeypatch.setattr(
        "app.language_detection.detector.translate_text_to_english",
        fake_translate,
    )

    result = identify_language_and_translate_to_english("நாளைக்கு சந்திப்பு")

    assert result["language"] == "ta"
    assert result["language_name"] == "Tamil"
    assert result["translated_text"] == "Meet tomorrow"
    assert result["translation_applied"] is True


def test_non_text_region_returns_unconfigured_result():
    result = LanguageDetector().detect_language(object())

    assert result.language == "unknown"
    assert result.confidence == 0.0
    assert result.metadata["reason"] == "image_region_detection_not_configured"



def test_translate_text_same_language_is_noop():
    assert translate_text("hello", "en", "en") == "hello"
