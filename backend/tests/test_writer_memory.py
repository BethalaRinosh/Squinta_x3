from app.writer_memory import DocumentLocalWriterMemory


def test_document_local_writer_memory_recommends_document_terms_without_overriding_visual_evidence():
    memory = DocumentLocalWriterMemory()
    memory.register_document(
        42,
        [
            "Karthik",
            "Karthik submitted the report",
            "Karthik met the team",
        ],
    )

    suggestions = memory.suggest_candidates("Karthlk")
    assert suggestions
    assert "Karthik" in suggestions

    decision = memory.evaluate_candidate("Karthlk", "Karthik", visual_support=0.7, risk_margin=0.2)
    assert decision["allowed"] is True
    assert decision["source"] == "document_memory"
    assert decision["visual_guard"] is True
