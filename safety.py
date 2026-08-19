"""Day 4 safety layer: claim extraction, verification, faithfulness, citation accuracy.

Wraps the Day 3 KidneyRAGGenerator output with a post-hoc verification pass and
computes the three Day 4 metrics the judges will ask for by name:

  * Retrieval Precision@k  — already in evaluate.py (Day 2)
  * Citation Accuracy       — Layer 1 (cited chunk_ids exist in retrieved set)
                              Layer 2 (cited evidence actually supports the claim)
  * Faithfulness            — (supported claims) / (total claims)

Also exposes the 4-level uncertainty language from the Day 4 PDF (strong /
partial / weak / insufficient) mapped from top-hit cosine similarity, and a
lightweight refusal-phrase helper for the frontend.

Design notes
------------
* The claim extractor is intentionally cheap: sentence-splitting the
  Recommendation section of the answer, filtered to substantive sentences.
  An LLM-based extractor is offered for parity with the Day 4 template, but
  the default is deterministic so evaluation is reproducible offline.
* The verifier ships in three flavours (LLM / embedding similarity / NLI
  cross-encoder). LLM is the primary path per Day 4 slide 14. Similarity is
  the offline fallback used when no API key is set (so tests + CI stay
  hermetic). NLI is available as an opt-in sanity check.
* Verification prompts are deliberately DIFFERENT from the generation prompt:
  the verifier is told its job is NOT to answer the question, only to judge
  claim ↔ evidence support. (Day 4 PDF module 1 / slides 44–45 of the notebook.)
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

# --- 4-level uncertainty language (Day 4 PDF slide 18) --------------------

MIN_TOP_COSINE = 0.70          # must stay in sync with generation.MIN_TOP_COSINE
WEAK_UPPER = 0.75
PARTIAL_UPPER = 0.85

EVIDENCE_LABELS = ("insufficient", "weak", "partial", "strong")

UNCERTAINTY_PHRASES = {
    "strong":   "The guideline recommends",
    "partial":  "The guideline suggests, though it doesn't directly address every detail of",
    "weak":     "Limited evidence found; consider consulting the full guideline on",
    "insufficient": None,      # refusal path — no phrase, use build_refusal()
}


def evidence_strength_from_cosine(cosine_sim: float | None,
                                  threshold: float = MIN_TOP_COSINE) -> str:
    """Map top-hit cosine similarity to a 4-level evidence-strength label.

    < threshold          -> insufficient (refuse)
    [threshold, 0.75)    -> weak         (hedge heavily)
    [0.75, 0.85)         -> partial      (state limitation)
    >= 0.85              -> strong       (direct grounded wording)
    """
    if cosine_sim is None or cosine_sim < threshold:
        return "insufficient"
    if cosine_sim < WEAK_UPPER:
        return "weak"
    if cosine_sim < PARTIAL_UPPER:
        return "partial"
    return "strong"


def uncertainty_phrase(level: str) -> str | None:
    """Return the recommended lead phrase for a given evidence-strength level."""
    return UNCERTAINTY_PHRASES.get(level)


# --- Claim extraction -----------------------------------------------------

# Captures a full citation of the form used by generation.format_citation():
#   [Source: <doc> — <section>, p.<N> | chunk_id:<id> | <url>]
CITATION_RE = re.compile(
    r"\[Source:\s*(?P<doc>[^—]+?)\s*—\s*(?P<section>.+?),\s*"
    r"(?P<pages>pp?\.\d+(?:[–-]\d+)?)\s*\|\s*chunk_id:(?P<chunk_id>\S+?)\s*\|\s*"
    r"(?P<url>[^\]]+?)\]"
)

# Split answer into sentences. Splits on .!? boundaries AND on hard newlines
# so that "### 1. Recommendation\nACR ≥30..." doesn't glue the header onto
# the first factual claim.
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

# Whole-line headers the model emits under our recommendation/excerpt/citation
# schema — never a factual claim. Anchored to strip an entire line, not just
# a prefix.
HEADER_LINE_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?\d+\.\s*(recommendation|excerpt|citation)s?\s*:?\s*$",
    re.IGNORECASE,
)

# Bare-word header (no numbering), e.g. "Recommendation:" — also skip.
BARE_HEADER_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(recommendation|excerpt|citation)s?\s*:?\s*$",
    re.IGNORECASE,
)

DISCLAIMER_MARKER = "---"

# Marker of a model-emitted refusal (Refusal Conditions 1–6 in the system
# prompt). If the answer starts with this line, there is no factual claim
# to verify — treat it as a refusal, not a set of claims.
REFUSAL_LEAD = "I can't answer this from the available sources."


def is_model_refusal(text: str) -> bool:
    """True when the LLM produced a refusal-shaped response instead of a claim.

    The generator returns refused=True only for gate-level refusals. But the
    system prompt also asks the model itself to refuse on 6 conditions, and
    those come back with refused=False, format_valid=False. We treat both as
    refusals for scoring purposes so faithfulness isn't computed on
    non-claim text.
    """
    if not text:
        return False
    return text.strip().startswith(REFUSAL_LEAD)


@dataclass
class Claim:
    text: str
    cited_chunk_ids: list[str] = field(default_factory=list)


def _strip_disclaimer(text: str) -> str:
    """Drop the trailing clinical disclaimer added by generation.py."""
    idx = text.rfind(DISCLAIMER_MARKER)
    return text[:idx] if idx != -1 else text


def _looks_like_claim(sentence: str) -> bool:
    stripped = sentence.strip()
    if len(stripped.split()) < 5:
        return False
    if HEADER_LINE_RE.match(stripped) or BARE_HEADER_RE.match(stripped):
        return False
    if stripped.startswith("[Source:") and stripped.endswith("]"):
        return False
    return True


def _strip_headers(text: str) -> str:
    """Remove whole lines that are just structural headers (Recommendation etc.)."""
    kept = []
    for line in text.splitlines():
        if HEADER_LINE_RE.match(line) or BARE_HEADER_RE.match(line):
            continue
        kept.append(line)
    return "\n".join(kept)


def extract_claims(answer: str) -> list[Claim]:
    """Split the answer into atomic factual claims + attach cited chunk_ids.

    Cited chunks are pulled from any [Source: ... | chunk_id:XXX | ...] tag
    in the whole answer body — a claim in the Recommendation and the
    citation in the Citation section are semantically linked even though
    they sit in different paragraphs.
    """
    if not answer:
        return []

    body = _strip_disclaimer(answer)
    cited = [m.group("chunk_id") for m in CITATION_RE.finditer(body)]

    # Drop headers and citation tags before sentence splitting so neither
    # leaks into a "claim".
    text = _strip_headers(body)
    text = CITATION_RE.sub("", text)

    claims: list[Claim] = []
    for raw in SENTENCE_SPLIT_RE.split(text):
        sentence = raw.strip().strip("\"'")
        if _looks_like_claim(sentence):
            claims.append(Claim(text=sentence, cited_chunk_ids=list(cited)))
    return claims


# --- Verification: embedding similarity (offline fallback) ----------------

def verify_claim_similarity(claim: str,
                            evidence: str,
                            embed_model,
                            threshold: float = 0.55) -> tuple[str, float]:
    """Cheap offline verifier: cosine similarity between claim and evidence.

    NOT logical entailment — it's a floor, useful when no LLM is available
    (tests, CI, offline demos). The threshold 0.55 was picked from the Day 4
    template's default; it's conservative for MedEmbed's 1024-d space.
    """
    import numpy as np
    vecs = embed_model.encode(
        [claim, evidence], normalize_embeddings=True, convert_to_numpy=True,
    )
    score = float(np.dot(vecs[0], vecs[1]))
    status = "SUPPORTED" if score >= threshold else "UNSUPPORTED"
    return status, score


# --- Verification: LLM verifier (primary) ---------------------------------

VERIFICATION_SYSTEM_PROMPT = (
    "You are an evidence-verification system. Your task is NOT to answer the "
    "user's question and NOT to add clinical judgment.\n\n"
    "Given a CLAIM and RETRIEVED EVIDENCE, determine whether the evidence "
    "directly supports the claim.\n\n"
    "Rules:\n"
    "1. Use ONLY the supplied evidence.\n"
    "2. Do not use outside medical knowledge.\n"
    "3. Do not infer facts not explicitly stated.\n"
    "4. If the evidence directly supports the claim, output SUPPORTED.\n"
    "5. Otherwise output UNSUPPORTED.\n"
    "6. Output exactly one token: SUPPORTED or UNSUPPORTED. No prose."
)


def verify_claim_llm(claim: str, evidence: str, pool,
                     tier_order: list[str] | None = None,
                     exclude_provider_ids: set[str] | None = None,
                     ) -> tuple[str, str]:
    """Ask an LLM (via LLMPool) whether the evidence supports the claim.

    Returns (status, provider_id). `status` is "SUPPORTED" or "UNSUPPORTED".

    `tier_order` defaults to `DEFAULT_VERIFY_TIER_ORDER` from llm_pool
    (HF first, Gemini second) — verification is a one-token classifier, so
    Qwen 7B handles it fine and we save Gemini quota for judge-facing answers.

    `exclude_provider_ids` lets the caller skip specific providers for this
    call (e.g. avoid burning the same Gemini key that just produced the
    answer we're now verifying).
    """
    from llm_pool import DEFAULT_VERIFY_TIER_ORDER
    prompt = f"CLAIM:\n{claim}\n\nRETRIEVED EVIDENCE:\n{evidence}"
    text, provider = pool.generate(
        system_prompt=VERIFICATION_SYSTEM_PROMPT,
        user_message=prompt,
        max_tokens=8,
        tier_order=tier_order or DEFAULT_VERIFY_TIER_ORDER,
        exclude_provider_ids=exclude_provider_ids,
    )
    raw = (text or "").strip().upper()
    # Robust to "UNSUPPORTED" starting with "SUPPORTED" as a substring.
    status = "SUPPORTED" if raw.startswith("SUPPORTED") else "UNSUPPORTED"
    return status, provider.id





# --- Verification: NLI cross-encoder (optional sanity check) --------------

_NLI_CACHE: dict[str, Any] = {}


def _load_nli(model_name: str = "cross-encoder/nli-deberta-v3-base"):
    if model_name not in _NLI_CACHE:
        from sentence_transformers import CrossEncoder
        _NLI_CACHE[model_name] = CrossEncoder(model_name)
    return _NLI_CACHE[model_name]


def verify_claim_nli(claim: str, evidence: str,
                     model_name: str = "cross-encoder/nli-deberta-v3-base",
                     entailment_threshold: float = 0.5,
                     ) -> tuple[str, float]:
    """Explicit textual entailment. Expensive; call only if you want the ablation."""
    import numpy as np
    model = _load_nli(model_name)
    scores = model.predict([(evidence, claim)])
    # cross-encoder/nli-deberta-v3-base label order: [contradiction, entailment, neutral]
    entail = float(scores[0][1]) if hasattr(scores[0], "__len__") else float(scores[1])
    status = "SUPPORTED" if entail >= entailment_threshold else "UNSUPPORTED"
    return status, entail


# --- Composite: score a full generator answer -----------------------------

@dataclass
class ClaimVerdict:
    claim: str
    status: str                   # SUPPORTED | UNSUPPORTED
    method: str                   # "llm" | "similarity" | "nli"
    score: float | None
    cited_chunk_ids: list[str]
    citations_valid: bool         # every cited chunk_id was in the retrieved set


@dataclass
class SafetyReport:
    evidence_strength: str        # strong | partial | weak | insufficient
    total_claims: int
    supported_claims: int
    faithfulness: float           # supported / total (NaN if 0 claims)
    citation_accuracy: float      # correct cited chunks / total cited chunks (NaN if 0)
    unsupported_claims: list[str]
    verdicts: list[ClaimVerdict]

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_strength": self.evidence_strength,
            "total_claims": self.total_claims,
            "supported_claims": self.supported_claims,
            "faithfulness": self.faithfulness,
            "citation_accuracy": self.citation_accuracy,
            "unsupported_claims": self.unsupported_claims,
            "verdicts": [v.__dict__ for v in self.verdicts],
        }


def _citation_accuracy(verdicts: list[ClaimVerdict]) -> float:
    total_cites = sum(len(v.cited_chunk_ids) for v in verdicts)
    if total_cites == 0:
        return float("nan")
    # Layer 1 (existence) is enforced by citations_valid, Layer 2 (support)
    # is enforced by status==SUPPORTED. A cited chunk is "correct" only if
    # both hold for its owning claim.
    correct = 0
    for v in verdicts:
        if not v.cited_chunk_ids:
            continue
        if v.citations_valid and v.status == "SUPPORTED":
            correct += len(v.cited_chunk_ids)
    return correct / total_cites


def score_answer(answer_text: str,
                 retrieved_hits: list[dict[str, Any]],
                 verifier: str = "similarity",
                 pool=None,
                 verify_tier_order: list[str] | None = None,
                 exclude_provider_ids: set[str] | None = None,
                 embed_model=None,
                 similarity_threshold: float = 0.55,
                 ) -> SafetyReport:
    """Score one generator answer against the chunks that produced it.

    `verifier` picks the verification method:
      * "similarity" — cheap, offline, deterministic. Needs embed_model.
      * "llm"        — primary. Needs pool (LLMPool from llm_pool).
      * "nli"        — cross-encoder. Loads a model on first call.

    For "llm" verification: `verify_tier_order` overrides the pool's tier
    order for verify calls only (default HF-first from llm_pool). Pass
    `exclude_provider_ids={generator.last_provider.id}` to keep verification
    off the same key that just answered.

    Evidence strength comes from the top hit's cosine_sim.
    """
    top_cosine = retrieved_hits[0].get("cosine_sim") if retrieved_hits else None
    strength = evidence_strength_from_cosine(top_cosine)

    if is_model_refusal(answer_text):
        # No claims to verify. Return a report with the refusal marker so
        # the caller can distinguish "no claims because refused" from
        # "no claims because parser failed".
        return SafetyReport(
            evidence_strength=strength,
            total_claims=0, supported_claims=0,
            faithfulness=float("nan"),
            citation_accuracy=float("nan"),
            unsupported_claims=[], verdicts=[],
        )

    claims = extract_claims(answer_text)
    if not claims:
        return SafetyReport(
            evidence_strength=strength,
            total_claims=0, supported_claims=0,
            faithfulness=float("nan"),
            citation_accuracy=float("nan"),
            unsupported_claims=[], verdicts=[],
        )

    retrieved_ids = {h["chunk_id"] for h in retrieved_hits}
    combined_evidence = "\n\n".join(h["text"] for h in retrieved_hits)

    verdicts: list[ClaimVerdict] = []
    for claim in claims:
        citations_valid = (not claim.cited_chunk_ids) or all(
            cid in retrieved_ids for cid in claim.cited_chunk_ids
        )
        if verifier == "llm":
            if pool is None:
                raise ValueError("verifier='llm' requires a pool (LLMPool)")
            status, _provider_id = verify_claim_llm(
                claim.text, combined_evidence, pool,
                tier_order=verify_tier_order,
                exclude_provider_ids=exclude_provider_ids,
            )
            score = None
        elif verifier == "nli":
            status, score = verify_claim_nli(claim.text, combined_evidence)
        else:
            if embed_model is None:
                raise ValueError("verifier='similarity' requires embed_model")
            status, score = verify_claim_similarity(
                claim.text, combined_evidence, embed_model, similarity_threshold,
            )
        verdicts.append(ClaimVerdict(
            claim=claim.text, status=status, method=verifier, score=score,
            cited_chunk_ids=claim.cited_chunk_ids, citations_valid=citations_valid,
        ))

    supported = sum(1 for v in verdicts if v.status == "SUPPORTED")
    faithfulness = supported / len(verdicts)
    unsupported = [v.claim for v in verdicts if v.status != "SUPPORTED"]

    return SafetyReport(
        evidence_strength=strength,
        total_claims=len(verdicts),
        supported_claims=supported,
        faithfulness=faithfulness,
        citation_accuracy=_citation_accuracy(verdicts),
        unsupported_claims=unsupported,
        verdicts=verdicts,
    )
