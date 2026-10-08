from app.language_detection.contextual_translation import find_protected_names, translate_with_context


def test_company_name_that_is_an_ordinary_word_is_protected():
    text = "My company's name is FISH, and we build software."
    assert find_protected_names(text) == ["FISH"]


def test_ordinary_fish_is_not_protected_without_name_context():
    assert find_protected_names("I like to eat fish.") == []


def test_brand_name_is_restored_after_translation():
    calls = []

    def fake_translator(text, source, target):
        calls.append((text, source, target))
        assert "FISH" not in text
        assert "ZXQPROTECTEDNAME0QXZ" in text
        return "मेरी कंपनी का नाम ZXQPROTECTEDNAME0QXZ है।"

    result = translate_with_context(
        "My company's name is FISH.", "en", "hi", fake_translator
    )

    assert result == "मेरी कंपनी का नाम FISH है।"
    assert len(calls) == 1


def test_unrelated_text_keeps_existing_translation_path():
    calls = []

    def fake_translator(text, source, target):
        calls.append((text, source, target))
        return "मुझे मछली पसंद है।"

    result = translate_with_context("I like fish.", "en", "hi", fake_translator)

    assert result == "मुझे मछली पसंद है।"
    assert calls == [("I like fish.", "en", "hi")]
