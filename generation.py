"""Day 3 grounded generation for Kidney-RAG.

Wires HybridRetriever.hybrid_search() output into a strictly grounded
recommendation/excerpt/citation answer, enforcing the retrieval-quality
gate and citation schema in code (not just in the prompt) so a weak
retrieval never reaches the model as if it were usable context.

Usage:
    from retrieval import HybridRetriever
    from generation import KidneyRAGGenerator

    retriever = HybridRetriever()
    generator = KidneyRAGGenerator(retriever)
    result = generator.answer("What is the diagnostic threshold for albuminuria in CKD?")
    print(result.text)
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from llm_pool import LLMPool, Provider, LLMPoolExhausted

load_dotenv()

SYSTEM_PROMPT_PATH = Path(__file__).parent / "kidney_rag_system_prompt.md"
MAX_TOKENS = 2000

# --- Retrieval-quality gate ------------------------------------------------
# Calibrated against eval/eval_set.json (18 questions, 2026-08-18):
#   In-scope  min cosine_sim = 0.7377 (q10: "Which drug class is first-line...")
#   OOS       max cosine_sim = 0.6330 (q18: "acute myocardial infarction")
#   Gap = 0.1047 → threshold 0.70 gives perfect separation on eval set.
MIN_TOP_COSINE = 0.70

# Confidence bands (cosine_sim of top hit)
CONFIDENCE_HIGH_THRESHOLD = 0.80
# Medium = [MIN_TOP_COSINE, CONFIDENCE_HIGH_THRESHOLD)
# Below MIN_TOP_COSINE → refuse

QUERY_LOG_DIR = Path(__file__).parent / "logs"
QUERY_LOG_PATH = QUERY_LOG_DIR / "query_log.jsonl"

CLINICAL_DISCLAIMER = (
    "\n\n---\n*This information is retrieved from indexed clinical guidelines "
    "and is not a substitute for professional medical advice. Always consult "
    "a qualified clinician for patient-specific decisions.*"
)


@dataclass
class GateResult:
    passed: bool
    reason: str | None = None
    confidence: str = "insufficient"
    used_hits: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class GenerationResult:
    text: str
    refused: bool
    gate_reason: str | None
    confidence: str
    hits_used: list[dict[str, Any]]
    format_valid: bool | None = None


def load_system_prompt(path: Path = SYSTEM_PROMPT_PATH) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"System prompt not found at {path}. Save kidney_rag_system_prompt.md "
            "next to generation.py, or pass an explicit path to load_system_prompt()."
        )
    return path.read_text(encoding="utf-8")


def quality_gate(hits: list[dict[str, Any]],
                 min_cosine: float = MIN_TOP_COSINE) -> GateResult:
    """Decide whether hybrid_search() output is strong enough to answer from.

    Uses cosine_sim of the top hit rather than fused_score. Calibration
    showed fused_score has no separation gap between in-scope and OOS,
    while cosine_sim has a 0.10+ gap at threshold 0.70.
    """
    if not hits:
        return GateResult(passed=False, reason="no relevant documents retrieved")

    top = hits[0]
    cosine_sim = top.get("cosine_sim")

    if cosine_sim is None or cosine_sim < min_cosine:
        return GateResult(passed=False, reason="no relevant documents retrieved")

    if cosine_sim >= CONFIDENCE_HIGH_THRESHOLD:
        confidence = "high"
    else:
        confidence = "medium"

    return GateResult(passed=True, confidence=confidence, used_hits=hits)


def format_citation(hit: dict[str, Any]) -> str:
    """Build the fixed-schema citation string from a hybrid_search() hit dict.

    [Source: <document_name> — <section_title>, p.<page> | chunk_id:<id> | <url>]
    """
    document_name = hit["document_name"]
    section_title = hit["section_title"] or "n/a"
    chunk_id = hit["chunk_id"]
    source_url = hit["source_url"]

    page_range = hit.get("page_range") or [hit["page_number"], hit["page_number"]]
    if page_range and page_range[0] != page_range[1]:
        page_part = f"pp.{page_range[0]}–{page_range[1]}"
    else:
        page_part = f"p.{hit['page_number']}"

    return (
        f"[Source: {document_name} — {section_title}, {page_part} "
        f"| chunk_id:{chunk_id} | {source_url}]"
    )


def format_context_block(hits: list[dict[str, Any]]) -> str:
    """Render retrieved chunks as labeled, citation-tagged context for the model."""
    blocks = []
    for i, hit in enumerate(hits, start=1):
        citation = format_citation(hit)
        blocks.append(
            f"--- Retrieved chunk {i} (chunk_id: {hit['chunk_id']}) ---\n"
            f"CITATION_STRING: {citation}\n"
            f"TEXT:\n{hit['text'].strip()}\n"
        )
    return "\n".join(blocks)


_CITATION_RE = re.compile(
    r"\[Source:\s*.+?\s*—\s*.+?,\s*pp?\.\d+.*?\|\s*chunk_id:\S+\s*\|.*?\]"
)


def validate_output_format(text: str) -> bool:
    """Check that LLM output follows recommendation/excerpt/citation structure."""
    lower = text.lower()
    has_recommendation = "recommendation" in lower
    has_excerpt = "excerpt" in lower
    has_citation = bool(_CITATION_RE.search(text))
    return has_recommendation and has_excerpt and has_citation


def build_refusal(reason: str) -> str:
    what_i_can_do = {
        "no relevant documents retrieved": (
            "try rephrasing toward a specific CKD guideline topic, or ask about "
            "one of the retrieved-but-unused passages if one seems close."
        ),
        "retrieved content lacks this specific detail": (
            "answer the general topic if you drop the specific numeric detail, "
            "or point you to the closest related passage."
        ),
        "this request requires clinical judgment or diagnosis beyond document lookup": (
            "share what the guidelines say about this topic in general terms, "
            "but please consult a qualified clinician for advice about your own care."
        ),
        "sources conflict without resolution": (
            "show you both conflicting passages side by side so you can see the discrepancy."
        ),
        "source currency cannot be confirmed": (
            "answer if you can confirm which guideline edition/version you need."
        ),
    }
    return (
        "I can't answer this from the available sources.\n"
        f"Reason: {reason}\n"
        f"What I can do: {what_i_can_do.get(reason, 'try a more specific question.')}"
    )


def _log_query(query: str, result: "GenerationResult", backend: str, model: str) -> None:
    """Append a structured log entry for every query to logs/query_log.jsonl."""
    QUERY_LOG_DIR.mkdir(exist_ok=True)

    top_hit = result.hits_used[0] if result.hits_used else None
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "backend": backend,
        "model": model,
        "gate_passed": not result.refused,
        "gate_reason": result.gate_reason,
        "confidence": result.confidence,
        "format_valid": result.format_valid,
        "top_cosine_sim": round(top_hit["cosine_sim"], 4) if top_hit else None,
        "top_chunk_id": top_hit["chunk_id"] if top_hit else None,
        "top_document": top_hit["document_name"] if top_hit else None,
        "top_section": top_hit["section_title"] if top_hit else None,
        "chunks_used": [
            {
                "chunk_id": h["chunk_id"],
                "cosine_sim": round(h["cosine_sim"], 4),
                "document_name": h["document_name"],
                "section_title": h["section_title"],
                "page_number": h["page_number"],
            }
            for h in result.hits_used
        ],
        "refused": result.refused,
        "answer_length": len(result.text),
        "answer_preview": result.text[:300],
    }

    with open(QUERY_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


class KidneyRAGGenerator:
    """Grounded generator backed by an LLMPool.

    The pool handles multi-key round-robin and cross-tier failover, so this
    class only worries about retrieval, gating, prompt assembly, format
    validation, and logging. The `last_provider` attribute records which
    concrete key served the most recent answer — useful for the UI badge
    and for `safety.verify_claim_llm` to avoid the same key on follow-ups.
    """

    def __init__(self, retriever, pool: LLMPool | None = None,
                 top_k: int = 5,
                 system_prompt_path: Path = SYSTEM_PROMPT_PATH):
        self.retriever = retriever
        self.top_k = top_k
        self.system_prompt = load_system_prompt(system_prompt_path)
        self.pool = pool or LLMPool.from_env()
        if not self.pool.has_providers():
            raise ValueError(
                "LLMPool has no providers. Set at least one of GOOGLE_API_KEY, "
                "HF_TOKEN, or ANTHROPIC_API_KEY in the environment."
            )
        self.last_provider: Provider | None = None

    # Legacy compatibility shims so older callers (evaluate scripts, tests,
    # logs) still see .backend and .model without knowing about the pool.
    @property
    def backend(self) -> str:
        p = self.last_provider or self.pool.providers()[0]
        return p.backend

    @property
    def model(self) -> str:
        p = self.last_provider or self.pool.providers()[0]
        return p.model

    def _call_llm(self, user_message: str) -> tuple[str, Provider]:
        text, provider = self.pool.generate(
            system_prompt=self.system_prompt,
            user_message=user_message,
            max_tokens=MAX_TOKENS,
        )
        self.last_provider = provider
        return text, provider

    def answer(self, query: str, k: int | None = None) -> GenerationResult:
        hits = self.retriever.hybrid_search(query, k=k or self.top_k)
        gate = quality_gate(hits)

        if not gate.passed:
            text = build_refusal(gate.reason)
            result = GenerationResult(
                text=text, refused=True, gate_reason=gate.reason,
                confidence="insufficient", hits_used=[], format_valid=None,
            )
            _log_query(query, result, "gate", "n/a")
            return result

        context_block = format_context_block(gate.used_hits)
        user_message = (
            f"Question: {query}\n\n"
            f"Retrieved context (use ONLY this; quote verbatim; reuse each "
            f"CITATION_STRING exactly as given):\n\n{context_block}"
        )

        text, provider = self._call_llm(user_message)
        format_valid = validate_output_format(text)
        text += CLINICAL_DISCLAIMER

        result = GenerationResult(
            text=text,
            refused=False,
            gate_reason=None,
            confidence=gate.confidence,
            hits_used=gate.used_hits,
            format_valid=format_valid,
        )
        _log_query(query, result, provider.backend, provider.model)
        return result


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    from retrieval import HybridRetriever

    retriever = HybridRetriever()
    generator = KidneyRAGGenerator(retriever)
    print(f"Pool providers: {[p.id for p in generator.pool.providers()]}")

    demo_questions = [
        "What is the diagnostic threshold for albuminuria in CKD?",
        "What is the recommended treatment for acute appendicitis?",
    ]
    for q in demo_questions:
        print("=" * 100)
        print(f"Q: {q}")
        result = generator.answer(q)
        who = generator.last_provider.id if generator.last_provider else "gate"
        print(f"[served_by={who} | confidence={result.confidence} | "
              f"refused={result.refused} | format_valid={result.format_valid}]")
        print(result.text)
        print()
