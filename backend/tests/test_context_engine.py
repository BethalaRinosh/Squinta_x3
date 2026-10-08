from app.context_engine import build_context, detect_domain, select_vocabulary


def test_medical_domain_detection():
    domain, confidence = detect_domain(
        "Patient has diabetes and hypertension. Start metformin 500 mg."
    )
    assert domain == "medical"
    assert confidence > 0


def test_finance_domain_detection():
    domain, confidence = detect_domain(
        "Revenue, expense, EBITDA and cash flow for Q4."
    )
    assert domain == "finance"
    assert confidence > 0


def test_general_fallback():
    domain, confidence = detect_domain("the quick brown fox jumps over the lazy dog")
    assert domain == "general"
    assert confidence == 0.0


def test_medical_context_contains_relevant_terms():
    context = build_context("Patient prescribed metformin 500 mg for diabetes")
    assert context["domain"] == "medical"
    terms = select_vocabulary(
        "Patient prescribed metformin 500 mg for diabetes",
        "medical",
    )
    assert "metformin" in terms or "diabetes" in terms
