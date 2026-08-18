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

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT_PATH = Path(__file__).parent / "kidney_rag_system_prompt.md"
MAX_TOKENS = 2000

# --- LLM backend selection ------------------------------------------------
# "huggingface": free (rate-limited ~1000 req/day), open-weight models
#                via HF Inference API. Default for this project.
# "anthropic":   paid, better instruction-following for strict format/refusal.
LLM_BACKEND = os.environ.get("KIDNEY_RAG_BACKEND", "huggingface")
ANTHROPIC_MODEL = "claude-sonnet-5"
HF_MODEL = os.environ.get("KIDNEY_RAG_HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
GEMINI_MODEL = os.environ.get("KIDNEY_RAG_GEMINI_MODEL", "gemini-3.6-flash")

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


class KidneyRAGGenerator:
    def __init__(self, retriever, api_key: str | None = None,
                 backend: str = LLM_BACKEND,
                 model: str | None = None, top_k: int = 5,
                 system_prompt_path: Path = SYSTEM_PROMPT_PATH):
        self.retriever = retriever
        self.backend = backend
        self.top_k = top_k
        self.system_prompt = load_system_prompt(system_prompt_path)

        if backend == "anthropic":
            from anthropic import Anthropic
            self.model = model or ANTHROPIC_MODEL
            self.client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        elif backend == "huggingface":
            from huggingface_hub import InferenceClient
            self.model = model or HF_MODEL
            token = api_key or os.environ.get("HF_TOKEN")
            if not token:
                raise ValueError("Set HF_TOKEN env var (or pass api_key=) for the huggingface backend.")
            self.client = InferenceClient(model=self.model, token=token)
        elif backend == "gemini":
            from google import genai
            self.model = model or GEMINI_MODEL
            self.client = genai.Client(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))
        else:
            raise ValueError(f"Unknown backend: {backend!r}. Use 'anthropic', 'huggingface', or 'gemini'.")

    def _call_llm(self, user_message: str) -> str:
        if self.backend == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=self.system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            return "".join(block.text for block in response.content if block.type == "text")

        if self.backend == "gemini":
            from google.genai import types
            response = self.client.models.generate_content(
                model=self.model,
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_prompt,
                    max_output_tokens=MAX_TOKENS,
                ),
            )
            return response.text

        completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=MAX_TOKENS,
        )
        return completion.choices[0].message.content

    def answer(self, query: str, k: int | None = None) -> GenerationResult:
        hits = self.retriever.hybrid_search(query, k=k or self.top_k)
        gate = quality_gate(hits)

        if not gate.passed:
            text = build_refusal(gate.reason)
            return GenerationResult(
                text=text, refused=True, gate_reason=gate.reason,
                confidence="insufficient", hits_used=[], format_valid=None,
            )

        context_block = format_context_block(gate.used_hits)
        user_message = (
            f"Question: {query}\n\n"
            f"Retrieved context (use ONLY this; quote verbatim; reuse each "
            f"CITATION_STRING exactly as given):\n\n{context_block}"
        )

        text = self._call_llm(user_message)
        format_valid = validate_output_format(text)
        text += CLINICAL_DISCLAIMER

        return GenerationResult(
            text=text,
            refused=False,
            gate_reason=None,
            confidence=gate.confidence,
            hits_used=gate.used_hits,
            format_valid=format_valid,
        )


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    from retrieval import HybridRetriever

    retriever = HybridRetriever()
    generator = KidneyRAGGenerator(retriever)

    demo_questions = [
        "What is the diagnostic threshold for albuminuria in CKD?",
        "What is the recommended treatment for acute appendicitis?",
    ]
    for q in demo_questions:
        print("=" * 100)
        print(f"Q: {q}")
        result = generator.answer(q)
        print(f"[confidence={result.confidence} | refused={result.refused} | format_valid={result.format_valid}]")
        print(result.text)
        print()
