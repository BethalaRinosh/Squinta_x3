from app.language_detection.contextual_translation import find_protected_names, translate_with_context


def test_possessive_company_name_is_preserved():
    source = "My company's name is FISH."
    assert find_protected_names(source) == ["FISH"]


def test_name_protection_does_not_freeze_regular_noun():
    assert find_protected_names("I like fish.") == []


def test_translation_masks_then_restores_company_name():
    seen = []
    def translator(text, source, target):
        seen.append(text)
        assert "FISH" not in text
        assert "ZXQPROTECTEDNAME0QXZ" in text
        return "मेरी कंपनी का नाम ZXQPROTECTEDNAME0QXZ है।"
    translated = translate_with_context("My company's name is FISH.", "en", "hi", translator)
    assert translated == "मेरी कंपनी का नाम FISH है।"
    assert len(seen) == 1
