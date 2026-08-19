"""Offline tests for safety.py (Day 4 layer).

No API calls, no model loads. Exercises the pure-Python pieces: 4-level
evidence-strength mapping, claim extractor, citation-accuracy math, and the
SafetyReport shape when a stub verifier is plugged in.
"""
from safety import (
    evidence_strength_from_cosine,
    uncertainty_phrase,
    extract_claims,
    score_answer,
    Claim,
    ClaimVerdict,
    _citation_accuracy,
)


SAMPLE_ANSWER = """### 1. Recommendation
ACR >= 30 mg/g defines clinically important albuminuria in adults with CKD. Category A3 corresponds to ACR > 300 mg/g.

### 2. Excerpt
"Albuminuria is defined as urinary albumin excretion >= 30 mg/24 hours (ACR >= 30 mg/g)."

### 3. Citation
[Source: KDIGO 2024 CKD Guideline — Definition of Albuminuria, p.15 | chunk_id:kdigo_p15_c20 | https://kdigo.org/]

---
*Clinical disclaimer.*
"""

RETRIEVED = [
    {"chunk_id": "kdigo_p15_c20", "cosine_sim": 0.87, "text": "Albuminuria is defined as ACR >= 30 mg/g."},
    {"chunk_id": "kdigo_p15_c21", "cosine_sim": 0.82, "text": "Category A3 = ACR > 300 mg/g severely increased."},
]


# --- evidence_strength_from_cosine ---------------------------------------

def test_strength_strong():
    assert evidence_strength_from_cosine(0.90) == "strong"
    assert evidence_strength_from_cosine(0.85) == "strong"


def test_strength_partial():
    assert evidence_strength_from_cosine(0.80) == "partial"
    assert evidence_strength_from_cosine(0.75) == "partial"


def test_strength_weak():
    assert evidence_strength_from_cosine(0.74) == "weak"
    assert evidence_strength_from_cosine(0.70) == "weak"


def test_strength_insufficient():
    assert evidence_strength_from_cosine(0.65) == "insufficient"
    assert evidence_strength_from_cosine(None) == "insufficient"
    assert evidence_strength_from_cosine(0.6999) == "insufficient"


def test_uncertainty_phrase_maps_all_levels():
    assert uncertainty_phrase("strong").startswith("The guideline recommends")
    assert "suggests" in uncertainty_phrase("partial")
    assert "Limited evidence" in uncertainty_phrase("weak")
    assert uncertainty_phrase("insufficient") is None


# --- extract_claims -------------------------------------------------------

def test_extractor_drops_headers():
    claims = extract_claims(SAMPLE_ANSWER)
    texts = [c.text for c in claims]
    assert not any(t.lower().startswith("recommendation") for t in texts)
    assert not any(t.lower().startswith("excerpt") for t in texts)
    assert not any(t.lower().startswith("citation") for t in texts)


def test_extractor_finds_recommendation_claims():
    claims = extract_claims(SAMPLE_ANSWER)
    texts = " ".join(c.text for c in claims)
    assert "ACR >= 30 mg/g" in texts
    assert "A3" in texts


def test_extractor_attaches_citations_across_paragraphs():
    claims = extract_claims(SAMPLE_ANSWER)
    assert all("kdigo_p15_c20" in c.cited_chunk_ids for c in claims)


def test_extractor_ignores_disclaimer_block():
    claims = extract_claims(SAMPLE_ANSWER)
    assert not any("disclaimer" in c.text.lower() for c in claims)


def test_extractor_handles_empty_answer():
    assert extract_claims("") == []
    assert extract_claims(None) == []


def test_extractor_skips_short_sentences():
    ans = "### 1. Recommendation\nYes. This is a longer sentence with more than five words in it."
    claims = extract_claims(ans)
    assert len(claims) == 1
    assert "longer sentence" in claims[0].text


# --- _citation_accuracy ---------------------------------------------------

def test_citation_accuracy_all_correct():
    v1 = ClaimVerdict(claim="c1", status="SUPPORTED", method="stub", score=None,
                      cited_chunk_ids=["a", "b"], citations_valid=True)
    v2 = ClaimVerdict(claim="c2", status="SUPPORTED", method="stub", score=None,
                      cited_chunk_ids=["c"], citations_valid=True)
    assert _citation_accuracy([v1, v2]) == 1.0


def test_citation_accuracy_partial():
    v1 = ClaimVerdict(claim="c1", status="SUPPORTED", method="stub", score=None,
                      cited_chunk_ids=["a", "b"], citations_valid=True)   # 2 correct
    v2 = ClaimVerdict(claim="c2", status="UNSUPPORTED", method="stub", score=None,
                      cited_chunk_ids=["c"], citations_valid=True)         # 0 correct
    v3 = ClaimVerdict(claim="c3", status="SUPPORTED", method="stub", score=None,
                      cited_chunk_ids=["d"], citations_valid=False)        # 0 correct
    # 2 of 4 total cites
    assert _citation_accuracy([v1, v2, v3]) == 0.5


def test_citation_accuracy_nan_when_no_citations():
    v = ClaimVerdict(claim="c", status="SUPPORTED", method="stub", score=None,
                     cited_chunk_ids=[], citations_valid=True)
    import math
    assert math.isnan(_citation_accuracy([v]))


# --- score_answer end-to-end with stub verifier --------------------------

class _StubEmbedModel:
    """Encodes claim + evidence to fixed high-similarity vectors so
    every claim comes back SUPPORTED. Lets us exercise score_answer without
    loading MedEmbed."""
    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
        import numpy as np
        # both vectors identical -> cosine = 1.0
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return np.stack([vec] * len(texts))


def test_score_answer_shape():
    report = score_answer(
        SAMPLE_ANSWER, RETRIEVED,
        verifier="similarity", embed_model=_StubEmbedModel(),
    )
    assert report.evidence_strength == "strong"     # top cosine 0.87
    assert report.total_claims >= 2
    assert report.supported_claims == report.total_claims
    assert report.faithfulness == 1.0
    assert report.citation_accuracy == 1.0
    assert report.unsupported_claims == []


def test_score_answer_flags_invalid_citation():
    bad_answer = SAMPLE_ANSWER.replace("kdigo_p15_c20", "kdigo_p999_c99")  # fabricated
    report = score_answer(
        bad_answer, RETRIEVED,
        verifier="similarity", embed_model=_StubEmbedModel(),
    )
    # Every verdict should have citations_valid=False
    assert all(not v.citations_valid for v in report.verdicts if v.cited_chunk_ids)
    # Citation accuracy must drop to zero (Layer 1 fail => nothing counts as correct)
    assert report.citation_accuracy == 0.0


def test_score_answer_insufficient_evidence_strength():
    weak_hits = [{"chunk_id": "x", "cosine_sim": 0.60, "text": "irrelevant"}]
    report = score_answer(
        SAMPLE_ANSWER, weak_hits,
        verifier="similarity", embed_model=_StubEmbedModel(),
    )
    assert report.evidence_strength == "insufficient"


def test_score_answer_empty_claims_returns_nan_metrics():
    import math
    empty_answer = "### 1. Recommendation\n### 2. Excerpt\n### 3. Citation\n"
    report = score_answer(
        empty_answer, RETRIEVED,
        verifier="similarity", embed_model=_StubEmbedModel(),
    )
    assert report.total_claims == 0
    assert math.isnan(report.faithfulness)
    assert math.isnan(report.citation_accuracy)


if __name__ == "__main__":
    import sys, inspect
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
        except Exception as e:
            failures += 1
            print(f"ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
