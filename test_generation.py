"""Offline tests for generation.py's gate + citation + format validation logic.

These don't touch Chroma, MedEmbed, or any LLM API — they exercise
quality_gate(), format_citation(), format_context_block(), build_refusal(),
and validate_output_format() directly against hand-built hit dicts shaped
like real hybrid_search() output.
"""
from generation import (
    quality_gate, format_citation, format_context_block,
    build_refusal, validate_output_format,
)

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
    "text": "Albuminuria is defined as urinary albumin excretion ≥30 mg/24 hours "
            "(or equivalent, ACR ≥30 mg/g).",
}

WEAK_HIT = {
    "rank": 1,
    "chunk_id": "kdigo2024_p9_001",
    "fused_score": 0.0009,
    "cosine_rank": None,
    "bm25_rank": 47,
    "cosine_sim": 0.55,
    "bm25_score": 0.4,
    "document_name": "KDIGO 2024 Clinical Practice Guideline for CKD",
    "section_title": "Foreword",
    "page_number": 9,
    "page_range": [9, 9],
    "source_url": "https://kdigo.org/guidelines/ckd-evaluation-and-management/",
    "text": "We thank the Work Group members for their dedication to this guideline.",
}

OOS_HIT = {
    "rank": 1,
    "chunk_id": "kdigo2024_p9_001",
    "fused_score": 0.0145,
    "cosine_rank": 38,
    "bm25_rank": 39,
    "cosine_sim": 0.63,
    "bm25_score": 0.2,
    "document_name": "KDIGO 2024 Clinical Practice Guideline for CKD",
    "section_title": "Foreword",
    "page_number": 9,
    "page_range": [9, 9],
    "source_url": "https://kdigo.org/guidelines/ckd-evaluation-and-management/",
    "text": "We thank the Work Group members for their dedication to this guideline.",
}

BORDERLINE_HIT = {**GOOD_HIT, "cosine_sim": 0.74, "chunk_id": "kdigo2024_p98_001"}

MULTI_PAGE_HIT = {**GOOD_HIT, "chunk_id": "kdigo2024_p88_010", "page_number": 88,
                  "page_range": [88, 89]}

NO_SECTION_HIT = {**GOOD_HIT, "chunk_id": "kdigo2024_p3_000", "section_title": ""}


# --- Gate tests ---

def test_gate_passes_on_strong_hit():
    gate = quality_gate([GOOD_HIT])
    assert gate.passed
    assert gate.confidence == "high"
    assert gate.used_hits == [GOOD_HIT]


def test_gate_passes_with_medium_confidence():
    gate = quality_gate([BORDERLINE_HIT])
    assert gate.passed
    assert gate.confidence == "medium"


def test_gate_fails_on_empty_results():
    gate = quality_gate([])
    assert not gate.passed
    assert gate.reason == "no relevant documents retrieved"
    assert gate.confidence == "insufficient"


def test_gate_fails_on_weak_cosine():
    gate = quality_gate([WEAK_HIT])
    assert not gate.passed
    assert gate.reason == "no relevant documents retrieved"


def test_gate_fails_on_oos_cosine():
    gate = quality_gate([OOS_HIT])
    assert not gate.passed


def test_gate_fails_on_none_cosine():
    no_cos = {**GOOD_HIT, "cosine_sim": None}
    gate = quality_gate([no_cos])
    assert not gate.passed


def test_gate_threshold_boundary():
    at_threshold = {**GOOD_HIT, "cosine_sim": 0.70}
    gate = quality_gate([at_threshold])
    assert gate.passed
    assert gate.confidence == "medium"

    below_threshold = {**GOOD_HIT, "cosine_sim": 0.6999}
    gate = quality_gate([below_threshold])
    assert not gate.passed


# --- Citation tests ---

def test_citation_single_page():
    citation = format_citation(GOOD_HIT)
    assert "kdigo2024_p42_003" in citation
    assert "p.42" in citation
    assert "KDIGO 2024 Clinical Practice Guideline for CKD" in citation
    assert "Definition and Classification of Albuminuria" in citation
    assert citation.startswith("[Source:") and citation.endswith("]")


def test_citation_page_range():
    citation = format_citation(MULTI_PAGE_HIT)
    assert "pp.88–89" in citation
    assert "p.88 " not in citation


def test_citation_missing_section_title():
    citation = format_citation(NO_SECTION_HIT)
    assert "n/a" in citation


def test_context_block_includes_citation_string_per_chunk():
    block = format_context_block([GOOD_HIT, MULTI_PAGE_HIT])
    assert block.count("CITATION_STRING:") == 2
    assert "chunk_id: kdigo2024_p42_003" in block
    assert "chunk_id: kdigo2024_p88_010" in block


# --- Refusal tests ---

def test_refusal_message_shape():
    msg = build_refusal("no relevant documents retrieved")
    assert msg.startswith("I can't answer this from the available sources.")
    assert "Reason: no relevant documents retrieved" in msg
    assert "What I can do:" in msg


def test_refusal_unknown_reason():
    msg = build_refusal("some unknown reason")
    assert "try a more specific question" in msg


# --- Output format validation tests ---

def test_validate_good_output():
    good = (
        "### 1. Recommendation\n"
        "ACR ≥30 mg/g defines albuminuria in CKD.\n\n"
        "### 2. Excerpt\n"
        "\"Albuminuria is defined as urinary albumin excretion ≥30 mg/24 hours\"\n\n"
        "### 3. Citation\n"
        "[Source: KDIGO 2024 — Definition, p.42 | chunk_id:kdigo2024_p42_003 | https://kdigo.org/]"
    )
    assert validate_output_format(good)


def test_validate_missing_recommendation():
    bad = (
        "### Excerpt\n\"some text\"\n"
        "### Citation\n[Source: X — Y, p.1 | chunk_id:z | http://x]"
    )
    assert not validate_output_format(bad)


def test_validate_missing_citation():
    bad = (
        "### 1. Recommendation\nSome rec.\n"
        "### 2. Excerpt\n\"Some quote.\"\n"
    )
    assert not validate_output_format(bad)


def test_validate_missing_excerpt():
    bad = (
        "### 1. Recommendation\nSome rec.\n"
        "### 3. Citation\n[Source: X — Y, p.1 | chunk_id:z | http://x]"
    )
    assert not validate_output_format(bad)


def test_validate_freeform_prose_fails():
    prose = "Based on the KDIGO guidelines, the threshold is 30 mg/g."
    assert not validate_output_format(prose)


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
