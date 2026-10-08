from app.config import settings
from app import ocr


def test_trocr_fallback_requires_local_cache_or_explicit_opt_in():
    assert settings.ENABLE_TROCR_FALLBACK is False
    assert ocr.should_try_trocr_fallback() is False
