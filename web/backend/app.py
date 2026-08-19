"""Kidney-RAG web backend.

Thin FastAPI layer that exposes the Day 3 grounded generator plus the Day 4
safety scoring over a JSON endpoint, and serves the static landing page
frontend from the same origin. Designed to be a single-process deployment
(uvicorn web.backend.app:app) so the app runs identically on localhost,
Render, Railway, or Fly.

Endpoints
---------
GET  /                         static index.html
GET  /health                   readiness probe + which backend is active
POST /api/ask   {question}     runs retrieval → gate → generation → safety
GET  /api/sources              catalog of indexed guidelines for the UI footer

Deliberate design choices:

* Retriever + generator are loaded ONCE on startup (each MedEmbed load is
  ~2s; each Chroma reload rescans the vectors). All requests reuse them.
* Every request answers within a single HTTP call — no streaming yet, since
  the safety pass has to see the full generated text before scoring
  faithfulness. If we add SSE later, safety runs after the stream closes.
* No auth. This is a demo. Deploy behind a URL you're happy sharing.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Reach into the project root so `retrieval` / `generation` / `safety` import
# cleanly whether we're launched from repo root or web/backend.
import sys as _sys
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

load_dotenv(_ROOT / ".env")

from retrieval import HybridRetriever            # noqa: E402
from generation import KidneyRAGGenerator, quality_gate, MIN_TOP_COSINE  # noqa: E402
from safety import (                              # noqa: E402
    score_answer, evidence_strength_from_cosine, uncertainty_phrase,
    is_model_refusal,
)
from llm_pool import LLMPool, LLMPoolExhausted    # noqa: E402

log = logging.getLogger("kidney_rag.web")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-5s %(name)s :: %(message)s")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

SOURCES = [
    {"name": "KDIGO 2024 CKD Guideline (full)", "pages": 199,
     "url": "https://kdigo.org/guidelines/ckd-evaluation-and-management/",
     "role": "Primary CKD management"},
    {"name": "NICE NG203 CKD", "pages": 78,
     "url": "https://www.nice.org.uk/guidance/ng203",
     "role": "UK diagnosis & pathway"},
    {"name": "KDIGO 2022 Diabetes Management in CKD", "pages": 128,
     "url": "https://kdigo.org/guidelines/diabetes-ckd/",
     "role": "SGLT2 / GLP-1 / finerenone"},
    {"name": "USPSTF CKD Screening", "pages": 6,
     "url": "https://www.uspreventiveservicestaskforce.org/",
     "role": "Screening (I-statement)"},
]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    k: int = Field(5, ge=1, le=10)


class Hit(BaseModel):
    chunk_id: str
    document_name: str
    section_title: str | None = None
    page_number: int
    page_range: list[int] | None = None
    source_url: str
    text: str
    cosine_sim: float | None
    fused_score: float | None
    rank: int


class SafetyPayload(BaseModel):
    evidence_strength: str
    total_claims: int
    supported_claims: int
    faithfulness: float | None
    citation_accuracy: float | None
    unsupported_claims: list[str]


class ProviderStatus(BaseModel):
    id: str
    backend: str
    key_env: str
    model: str
    tier: str
    status: str                     # "ready" | "cooldown"
    cooldown_remaining_s: float
    successes: int
    failures: int
    last_error: str | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    refused: bool
    refusal_kind: str | None                  # gate | model | null
    confidence: str                           # high | medium | insufficient
    evidence_strength: str                    # strong | partial | weak | insufficient
    uncertainty_lead: str | None
    top_cosine: float | None
    format_valid: bool | None
    served_by: str | None                     # provider id used for generation
    hits: list[Hit]
    safety: SafetyPayload | None


class HealthResponse(BaseModel):
    ok: bool
    retriever_ready: bool
    generator_ready: bool
    backend: str | None                       # first ready provider's backend
    model: str | None
    chunks_indexed: int
    embed_model: str
    min_top_cosine: float
    providers: list[ProviderStatus]


# --- lifespan: load MedEmbed + generator once, keep them warm --------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading HybridRetriever (MedEmbed + Chroma + BM25)…")
    retriever = HybridRetriever()
    log.info(f"Retriever ready. {retriever.collection.count()} chunks indexed.")

    # One shared pool for BOTH generation and verification. Different tier
    # orders per call site (see safety.verify_claim_llm) route work to the
    # right backend without needing two pools or two sets of keys.
    pool = LLMPool.from_env()
    if pool.has_providers():
        log.info("LLMPool ready with %d providers across %d tier(s): %s",
                 len(pool.providers()), len(pool.tier_order),
                 [p.id for p in pool.providers()])
    else:
        log.warning("LLMPool has no providers. /api/ask will return retrieval-only "
                    "refusal path. Set GOOGLE_API_KEY / HF_TOKEN / ANTHROPIC_API_KEY.")

    generator: KidneyRAGGenerator | None = None
    if pool.has_providers():
        try:
            generator = KidneyRAGGenerator(retriever, pool=pool)
            log.info("Generator ready (pool-backed).")
        except Exception as exc:
            log.warning("Generator init failed: %s: %s", type(exc).__name__, exc)

    app.state.retriever = retriever
    app.state.pool = pool
    app.state.generator = generator
    try:
        yield
    finally:
        log.info("Shutting down.")


app = FastAPI(
    title="Kidney-RAG",
    description="Retrieval-grounded clinical QA over 4 CKD guidelines "
                "(KDIGO 2024, KDIGO 2022 DM, NICE NG203, USPSTF).",
    version="0.4.0",  # Day 4
    lifespan=lifespan,
)

# CORS is permissive because the same origin serves the frontend and the API.
# Kept explicit so a split-origin deploy still works if you point the frontend
# somewhere else later.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    r = app.state.retriever
    g = app.state.generator
    pool: LLMPool = app.state.pool
    provider_rows = [ProviderStatus(**row) for row in pool.status()]
    return HealthResponse(
        ok=True,
        retriever_ready=r is not None,
        generator_ready=g is not None,
        backend=getattr(g, "backend", None) if g else None,
        model=getattr(g, "model", None) if g else None,
        chunks_indexed=r.collection.count(),
        embed_model=r.model_name,
        min_top_cosine=MIN_TOP_COSINE,
        providers=provider_rows,
    )


@app.get("/api/sources")
def sources() -> dict[str, Any]:
    r = app.state.retriever
    return {"corpus_size": r.collection.count(), "guidelines": SOURCES}


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    r = app.state.retriever
    g: KidneyRAGGenerator | None = app.state.generator

    hits_raw = r.hybrid_search(req.question, k=req.k)
    top_cos = hits_raw[0]["cosine_sim"] if hits_raw else None
    gate = quality_gate(hits_raw)
    strength = evidence_strength_from_cosine(top_cos)

    # Public hit payload — safe subset of the internal dict.
    public_hits = [Hit(
        chunk_id=h["chunk_id"],
        document_name=h["document_name"],
        section_title=h["section_title"],
        page_number=h["page_number"],
        page_range=h.get("page_range"),
        source_url=h["source_url"],
        text=h["text"],
        cosine_sim=h.get("cosine_sim"),
        fused_score=h.get("fused_score"),
        rank=h["rank"],
    ) for h in hits_raw]

    # Path 1: gate-level refusal (no LLM call, deterministic response).
    if not gate.passed:
        from generation import build_refusal, CLINICAL_DISCLAIMER
        refusal = build_refusal(gate.reason or "no relevant documents retrieved")
        return AskResponse(
            question=req.question,
            answer=refusal + CLINICAL_DISCLAIMER,
            refused=True,
            refusal_kind="gate",
            confidence="insufficient",
            evidence_strength=strength,
            uncertainty_lead=None,
            top_cosine=top_cos,
            format_valid=None,
            served_by=None,
            hits=public_hits,
            safety=None,
        )

    # Path 2: no generator available (e.g. no API key on the server).
    if g is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Generator not configured on this server (no LLM API key). "
                "Retrieval works — see the /api/ask response's `hits` — but "
                "no answer text can be produced. Set GOOGLE_API_KEY (Gemini), "
                "HF_TOKEN (HuggingFace), or ANTHROPIC_API_KEY."
            ),
        )

    # Path 3: full pipeline — generate + score.
    try:
        result = g.answer(req.question, k=req.k)
    except LLMPoolExhausted as exc:
        log.warning("LLM pool exhausted: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=(
                "All LLM providers are rate-limited or in cooldown. "
                "Retrieval still works — see the 'hits' the retriever surfaced. "
                "Please try again in ~1 minute."
            ),
        ) from exc
    except Exception as exc:  # network, auth, other provider error
        log.exception("Generation error")
        raise HTTPException(
            status_code=502,
            detail=f"Upstream LLM error: {type(exc).__name__}: {exc}",
        ) from exc

    refused = result.refused
    refusal_kind = "gate" if refused else ("model" if is_model_refusal(result.text) else None)
    served_by = g.last_provider.id if g.last_provider and not refused else None

    safety_payload: SafetyPayload | None = None
    if not refused and refusal_kind is None:
        # Only score the answer when there IS an answer. For refusals the
        # safety metrics aren't meaningful (no claims to verify).
        # Similarity verifier — cheap, offline, deterministic. Swap to
        # verifier="llm" + pool=g.pool if you want LLM-verified faithfulness
        # per request (costs 2-5 extra LLM calls per answer).
        report = score_answer(
            answer_text=result.text,
            retrieved_hits=hits_raw,
            verifier="similarity",
            embed_model=r.embed_model,
        )
        safety_payload = SafetyPayload(
            evidence_strength=report.evidence_strength,
            total_claims=report.total_claims,
            supported_claims=report.supported_claims,
            faithfulness=(None if report.faithfulness != report.faithfulness  # NaN check
                          else round(report.faithfulness, 4)),
            citation_accuracy=(None if report.citation_accuracy != report.citation_accuracy
                               else round(report.citation_accuracy, 4)),
            unsupported_claims=report.unsupported_claims,
        )

    return AskResponse(
        question=req.question,
        answer=result.text,
        refused=refused,
        refusal_kind=refusal_kind,
        confidence=result.confidence,
        evidence_strength=strength,
        uncertainty_lead=uncertainty_phrase(strength),
        top_cosine=top_cos,
        format_valid=result.format_valid,
        served_by=served_by,
        hits=public_hits,
        safety=safety_payload,
    )


# --- static frontend -------------------------------------------------------

# Mount /assets so styles.css / app.js / favicon.svg load with cache-friendly
# URLs. Root path returns index.html.

if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR)), name="assets")

    @app.get("/")
    def root() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")
else:
    log.warning(f"Frontend directory not found at {FRONTEND_DIR}. API-only mode.")
