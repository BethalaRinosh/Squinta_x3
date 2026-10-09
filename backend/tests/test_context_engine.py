from app.context_engine import (
    build_context,
    build_context_prompt,
    detect_domain,
    select_vocabulary,
)


def test_medical_domain_detection():
    domain, confidence = detect_domain(
        "Patient has diabetes and hypertension. Start metformin 500 mg."
    )
    assert domain == "medical"
    assert confidence >= 0.55


def test_finance_domain_detection():
    domain, confidence = detect_domain(
        "Revenue, expense, EBITDA and cash flow for Q4."
    )
    assert domain == "finance"
    assert confidence >= 0.55


def test_general_fallback():
    domain, confidence = detect_domain("the quick brown fox jumps over the lazy dog")
    assert domain == "general"
    assert confidence == 0.0


def test_single_weak_clue_does_not_force_a_domain():
    domain, confidence = detect_domain("patient")
    assert domain == "general"
    assert confidence == 0.0


def test_domain_terms_use_word_boundaries():
    domain, confidence = detect_domain("party started early")
    assert domain == "general"
    assert confidence == 0.0


def test_medical_context_contains_relevant_terms_and_evidence():
    text = "Patient prescribed metformin 500 mg for diabetes"
    context = build_context(text)
    assert context["domain"] == "medical"
    assert context["confidence"] >= 0.55
    assert "patient" in context["evidence"]
    terms = select_vocabulary(text, "medical")
    assert "metformin" in terms or "diabetes" in terms


def test_ambiguous_context_stays_general():
    context = build_context("question about the result")
    assert context["domain"] == "general"
    assert context["vocabulary"] == []


def test_forced_domain_is_explicit_and_reported():
    context = build_context("take this", forced_domain="medical")
    assert context["domain"] == "medical"
    assert context["forced"] is True
    assert context["confidence"] == 1.0


def test_context_prompt_warns_against_inventing_critical_values():
    text = "Patient has diabetes and hypertension. Start metformin 500 mg."
    context = build_context(text)
    prompt = build_context_prompt(text, context)
    assert "OBSERVED DOMAIN CLUES" in prompt
    assert "Do not infer missing words" in prompt
    assert "numbers" in prompt
