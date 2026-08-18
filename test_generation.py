"""Offline tests for generation.py's gate + citation logic.

These don't touch Chroma, MedEmbed, or the Anthropic API — they exercise
quality_gate(), format_citation(), and format_context_block() directly
against hand-built hit dicts shaped like real hybrid_search() output.
"""
from generation import quality_gate, format_citation, format_context_block, build_refusal

GOOD_HIT = {
    "rank": 1,
    "chunk_id": "kdigo2024_p42_003",
    "fused_score": 0.0114,
    "cosine_rank": 1,
    "bm25_rank": 2,
    "cosine_sim": 0.81,
    "bm25_score": 6.7,
    "document_name": "KDIGO 2024 Clinical Practice Guideline for CKD",
    "section_title": "Definition and Classification of Albuminuria",
    "page_number": 42,
    "page_range": [42, 42],
    "source_url": "https://kdigo.org/guidelines/ckd-evaluation-and-management/",
    "text": "Albuminuria is defined as urinary albumin excretion \u226530 mg/24 hours "
            "(or equivalent, ACR \u226530 mg/g).",
}

WEAK_HIT = {
    "rank": 1,
    "chunk_id": "kdigo2024_p9_001",
    "fused_score": 0.0009,
    "cosine_rank": None,
    "bm25_rank": 47,
    "cosine_sim": None,
    "bm25_score": 0.4,
    "document_name": "KDIGO 2024 Clinical Practice Guideline for CKD",
    "section_title": "Foreword",
    "page_number": 9,
    "page_range": [9, 9],
    "source_url": "https://kdigo.org/guidelines/ckd-evaluation-and-management/",
    "text": "We thank the Work Group members for their dedication to this guideline.",
}

MULTI_PAGE_HIT = {**GOOD_HIT, "chunk_id": "kdigo2024_p88_010", "page_number": 88,
                  "page_range": [88, 89]}

NO_SECTION_HIT = {**GOOD_HIT, "chunk_id": "kdigo2024_p3_000", "section_title": ""}


def test_gate_passes_on_strong_hit():
    gate = quality_gate([GOOD_HIT])
    assert gate.passed
    assert gate.used_hits == [GOOD_HIT]


def test_gate_fails_on_empty_results():
    gate = quality_gate([])
    assert not gate.passed
    assert gate.reason == "no relevant documents retrieved"


def test_gate_fails_on_weak_top_hit():
    gate = quality_gate([WEAK_HIT])
    assert not gate.passed
    assert gate.reason == "no relevant documents retrieved"


def test_gate_fails_on_no_provenance():
    no_prov = {**GOOD_HIT, "cosine_rank": None, "bm25_rank": None}
    gate = quality_gate([no_prov])
    assert not gate.passed


def test_citation_single_page():
    citation = format_citation(GOOD_HIT)
    assert "kdigo2024_p42_003" in citation
    assert "p.42" in citation
    assert "KDIGO 2024 Clinical Practice Guideline for CKD" in citation
    assert "Definition and Classification of Albuminuria" in citation
    assert citation.startswith("[Source:") and citation.endswith("]")


def test_citation_page_range():
    citation = format_citation(MULTI_PAGE_HIT)
    assert "pp.88\u201389" in citation
    assert "p.88 " not in citation  # must not fall back to single-page form


def test_citation_missing_section_title():
    citation = format_citation(NO_SECTION_HIT)
    assert "n/a" in citation


def test_context_block_includes_citation_string_per_chunk():
    block = format_context_block([GOOD_HIT, MULTI_PAGE_HIT])
    assert block.count("CITATION_STRING:") == 2
    assert "chunk_id: kdigo2024_p42_003" in block
    assert "chunk_id: kdigo2024_p88_010" in block


def test_refusal_message_shape():
    msg = build_refusal("no relevant documents retrieved")
    assert msg.startswith("I can't answer this from the available sources.")
    assert "Reason: no relevant documents retrieved" in msg
    assert "What I can do:" in msg


if __name__ == "__main__":
    import sys
    import inspect

    tests = [(name, fn) for name, fn in globals().items()
             if name.startswith("test_") and inspect.isfunction(fn)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {name}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
