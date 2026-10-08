from app.evidence_gate import EvidenceGate, EvidenceDecision


def test_evidence_gate_accepts_visually_supported_correction():
    gate = EvidenceGate()
    decision = gate.analyze(
        raw_text="tomorow",
        candidates=["tomorrow", "tomorow", "tomorrov"],
        image_quality=0.9,
        model_agreement=1.0,
    )

    assert isinstance(decision, EvidenceDecision)
    assert decision.decision in {"ACCEPT", "REVIEW"}
    assert decision.accepted_text in {"tomorrow", "tomorow"}
    assert decision.risk_score >= 0.0
    assert decision.risk_score <= 1.0


def test_evidence_gate_abstains_when_visual_support_is_low():
    gate = EvidenceGate()
    decision = gate.analyze(
        raw_text="xyzqq",
        candidates=["meeting", "m33ting", "zzzzzz"],
        image_quality=0.2,
        model_agreement=0.2,
    )

    assert decision.decision == "ABSTAIN"
    assert decision.accepted_text == "xyzqq"
