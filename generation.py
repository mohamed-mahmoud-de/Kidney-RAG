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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SYSTEM_PROMPT_PATH = Path(__file__).parent / "kidney_rag_system_prompt.md"
MAX_TOKENS = 1200

# --- LLM backend selection ---------------------------------------------
# "anthropic": paid, best instruction-following for the strict
#              recommendation/excerpt/citation + refusal rules.
# "huggingface": free (rate-limited, ~1000 req/day), open-weight models
#              via HF's OpenAI-compatible router. Weaker at following the
#              strict format/refusal rules — expect more manual QA.
LLM_BACKEND = os.environ.get("KIDNEY_RAG_BACKEND", "anthropic")
ANTHROPIC_MODEL = "claude-sonnet-4-6"
HF_MODEL = os.environ.get("KIDNEY_RAG_HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# --- Retrieval-quality gate config -----------------------------------------
# These are starting points (see the Day-2 notebook's alpha-sweep TODO to
# calibrate against a labeled eval set instead of intuition). RRF fused
# scores from weighted_rrf(w_semantic=0.7, w_lexical=0.3, k_rrf=60) are
# small numbers (~0.005-0.016 typical for a real top hit at k=60), so the
# floor below is deliberately conservative and should be tuned per corpus.
MIN_TOP_FUSED_SCORE = 0.008     # top hit must clear this to be "relevant"
MIN_PROVENANCE_HITS = 1         # top hit must appear in at least one of cos/bm25 ranks


@dataclass
class GateResult:
    passed: bool
    reason: str | None = None            # populated when passed is False
    used_hits: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class GenerationResult:
    text: str
    refused: bool
    gate_reason: str | None
    hits_used: list[dict[str, Any]]


def load_system_prompt(path: Path = SYSTEM_PROMPT_PATH) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"System prompt not found at {path}. Save kidney-rag-system-prompt.md "
            "next to generation.py, or pass an explicit path to load_system_prompt()."
        )
    return path.read_text(encoding="utf-8")


def quality_gate(hits: list[dict[str, Any]],
                  min_fused_score: float = MIN_TOP_FUSED_SCORE,
                  min_provenance: int = MIN_PROVENANCE_HITS) -> GateResult:
    """Decide whether hybrid_search() output is strong enough to answer from.

    This mirrors the "Retrieval-quality gate" section of the system prompt,
    but enforced in code: a returned hit is not the same as a relevant hit,
    since hybrid_search() always returns up to k rows regardless of quality.
    """
    if not hits:
        return GateResult(passed=False, reason="no relevant documents retrieved")

    top = hits[0]
    provenance = sum(1 for key in ("cosine_rank", "bm25_rank") if top.get(key) is not None)

    if top["fused_score"] < min_fused_score:
        return GateResult(passed=False, reason="no relevant documents retrieved")
    if provenance < min_provenance:
        return GateResult(passed=False, reason="no relevant documents retrieved")

    return GateResult(passed=True, used_hits=hits)


def format_citation(hit: dict[str, Any]) -> str:
    """Build the fixed-schema citation string from a hybrid_search() hit dict.

    [Source: <document_name> — <section_title>, p.<page_number> | chunk_id:<chunk_id> | <source_url>]
    """
    document_name = hit["document_name"]
    section_title = hit["section_title"] or "n/a"
    chunk_id = hit["chunk_id"]
    source_url = hit["source_url"]

    page_range = hit.get("page_range") or [hit["page_number"], hit["page_number"]]
    if page_range and page_range[0] != page_range[1]:
        page_part = f"pp.{page_range[0]}\u2013{page_range[1]}"
    else:
        page_part = f"p.{hit['page_number']}"

    return (
        f"[Source: {document_name} \u2014 {section_title}, {page_part} "
        f"| chunk_id:{chunk_id} | {source_url}]"
    )


def format_context_block(hits: list[dict[str, Any]]) -> str:
    """Render retrieved chunks as labeled, citation-tagged context for the model.

    The model must quote only from these blocks and must reuse the citation
    string verbatim rather than reconstruct it, which keeps the final output's
    citations byte-identical to what format_citation() would produce.
    """
    blocks = []
    for i, hit in enumerate(hits, start=1):
        citation = format_citation(hit)
        blocks.append(
            f"--- Retrieved chunk {i} (chunk_id: {hit['chunk_id']}) ---\n"
            f"CITATION_STRING: {citation}\n"
            f"TEXT:\n{hit['text'].strip()}\n"
        )
    return "\n".join(blocks)


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
        else:
            raise ValueError(f"Unknown backend: {backend!r}. Use 'anthropic' or 'huggingface'.")

    def _call_llm(self, user_message: str) -> str:
        if self.backend == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=self.system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            return "".join(block.text for block in response.content if block.type == "text")

        # huggingface: OpenAI-style chat completion via the router
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
            return GenerationResult(text=text, refused=True, gate_reason=gate.reason, hits_used=[])

        context_block = format_context_block(gate.used_hits)
        user_message = (
            f"Question: {query}\n\n"
            f"Retrieved context (use ONLY this; quote verbatim; reuse each "
            f"CITATION_STRING exactly as given):\n\n{context_block}"
        )

        text = self._call_llm(user_message)

        return GenerationResult(
            text=text,
            refused=False,
            gate_reason=None,
            hits_used=gate.used_hits,
        )


if __name__ == "__main__":
    from retrieval import HybridRetriever

    retriever = HybridRetriever()
    generator = KidneyRAGGenerator(retriever)

    demo_questions = [
        "What is the diagnostic threshold for albuminuria in CKD?",
        "What is the recommended treatment for acute appendicitis?",  # expect refusal
    ]
    for q in demo_questions:
        print("=" * 100)
        print(f"Q: {q}")
        result = generator.answer(q)
        print(result.text)
        print()
