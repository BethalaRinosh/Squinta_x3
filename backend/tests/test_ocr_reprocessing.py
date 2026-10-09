from app.routes.ocr import _can_start_ocr


def test_completed_page_can_be_explicitly_reprocessed():
    assert _can_start_ocr("done") is True


def test_processing_page_cannot_be_queued_twice():
    assert _can_start_ocr("processing") is False


def test_idle_and_error_pages_can_be_processed():
    assert _can_start_ocr("idle") is True
    assert _can_start_ocr("error") is True
    assert _can_start_ocr(None) is True
