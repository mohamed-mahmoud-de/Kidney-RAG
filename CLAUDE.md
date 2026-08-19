# Kidney-RAG — Project brief for Claude

> Read this first. Don't re-explore.

## What this is

Creativa × Orange AI Hackathon (Aug 16–20, 2026). A **clinical RAG** over official public CKD guidelines. Every clinical answer must trace to `document + section + page + chunk_id`. Core philosophy: **Fluent Answer ≠ Safe Answer.**

## Corpus (do not re-download)

| Doc | File | Pages | Role in index |
|-----|------|-------|---------------|
| KDIGO 2024 CKD Guideline (full) | `corpus/raw_pdfs/KDIGO_2024_CKD_Guideline_full.pdf` | 199 | **Indexed** — primary CKD management |
| NICE NG203 CKD | `corpus/raw_pdfs/NICE_NG203_CKD_assessment_and_management.pdf` | 78 | **Indexed** — UK diagnosis/pathway |
| KDIGO 2022 Diabetes Management in CKD | `corpus/raw_pdfs/KDIGO_2022_Diabetes_Management_in_CKD.pdf` | 128 | **Indexed** — SGLT2/GLP-1/finerenone deep source |
| USPSTF CKD Screening | `corpus/raw_pdfs/USPSTF_CKD_Screening_Recommendation.pdf` | 6 | **Indexed** — safe-refusal Case C (I-statement) |

The KDIGO Summary of Recs and Exec Summary PDFs were held out (they duplicate the full body) and were used as the answer key when building the eval set. They are no longer in `raw_pdfs/`.

## Pipeline

```
raw_pdfs/       pdf_parser.ipynb       chunker.ipynb        embedder.ipynb        retrieval.py
 KDIGO   ──▶ parsed/kdigo_parsed.json ──▶                ┐                   ┌─▶ HybridRetriever
 NICE    ──▶ parsed/nice_parsed.json  ──▶ chunks/        ├──▶ chunk_embeddings.npy
 KDIGO-DM ─▶ parsed/kdigo_diabetes... ──▶ all_chunks     │    chroma_db/     │
 USPSTF  ──▶ parsed/uspstf_parsed.json──▶    .jsonl      ┘  (ckd_guidelines) │
                                                                              │
                                                                              ▼
                                              evaluate.py / model_compare.py / ablation.py
                                                     ↓ read eval/eval_set.json
                                              artifacts/day2/*.{json,csv}
```

### Run each stage

Ingestion (rebuilds the index — only needed after corpus changes):
```bash
python -m jupyter nbconvert --to notebook --execute pdf_parser.ipynb --inplace
python -m jupyter nbconvert --to notebook --execute chunker.ipynb   --inplace
python -m jupyter nbconvert --to notebook --execute embedder.ipynb  --inplace   # ~15-20 min first run
```

Evaluation & retrieval sandbox (fast — assume the index exists):
```bash
python evaluate.py --sweep-k                    # Precision@k, Recall@k, Hit@k, OOS max cosine
python model_compare.py                         # MedEmbed vs BGE-large head-to-head
python ablation.py                              # chunk-size ablation (400/40, 550/55, 700/70)
python explain.py "your clinical question"     # CLI evidence view for one query
```

Notebooks are self-contained. Each reads its input, writes its output, and is idempotent. Always use `--inplace` (never `--output foo.ipynb`).

## Chunk contract (frozen — do not change)

```json
{"chunk_id": "kdigo_dm_p44_c02",
 "document_name": "KDIGO 2022 Diabetes Management in CKD",
 "source_url": "https://kdigo.org/...",
 "page_number": 44,
 "page_range": [44, 48],
 "section_title": "1.3 Sodium–glucose cotransporter-2 inhibitors",
 "token_count": 483,
 "text": "..."}
```

- `chunk_id` format: `<doc-short>_p<start_page>_c<index>` where doc-short is `kdigo`, `nice`, `kdigo_dm`, or `uspstf`.
- Chroma stores `page_start` / `page_end` as separate ints (Chroma metadata must be flat scalars — no lists).
- **666 chunks total** — KDIGO 340 + NICE 72 + KDIGO-DM 241 + USPSTF 13. Median 510 tokens, hard cap 750, 0 oversized.
- Chunks are dropped from the index when `is_clinical_section()` returns False (research agendas, author bios, notice, acknowledgments). See `chunker.ipynb` cell 1.
- KDIGO 2024 pp 34–53 and KDIGO 2022 pp 20–29 (Summary of Recs) are excluded via `SKIP_PAGES` — they duplicate the body.

## Stack decisions (locked, empirically validated)

- **Parser:** PyMuPDF (`fitz`). pdfplumber was tested — it glues words together on KDIGO's 2-column layout.
- **Tokenizer:** `tiktoken` `cl100k_base` (for chunk sizing; embedding model has its own tokenizer).
- **Embeddings:** `abhinand/MedEmbed-large-v0.1` (1024-dim, bge-large fine-tuned on PubMed).
  - Beat `BAAI/bge-large-en-v1.5` by **+14 pts Hit@5** on our eval set (`artifacts/day2/model_compare.json`).
  - Local via `sentence-transformers`. **No API. No rate limits.**
  - Query-side prefix: `"Represent this medical question for retrieving relevant clinical guideline passages: "` (bge-style).
  - L2-normalized → cosine similarity = dot product.
- **Vector DB:** ChromaDB persistent client at `corpus/chunks/chroma_db/`, cosine space, collection `ckd_guidelines`.
- **Retrieval:** hybrid semantic + BM25 fused with weighted Reciprocal Rank Fusion (RRF).
  - Weights: **0.7 semantic / 0.3 lexical**. RRF damping k=60. Candidate pool 50.
  - Ties semantic-only on Hit@5 (both 0.867), robust on threshold-anchored queries.
- **Top-k = 5.** Chosen because Hit@k plateaus there (`artifacts/day2/topk_curve.csv`).
- **Chunk config = 550/55.** Ablation showed 400/40 slightly ahead on page-level Hit@5, but scoring was a proxy (different embed model, page-level scoring) — flagged for Day-3+ validation.
- **HF model cache:** default `~/.cache/huggingface/hub/`. Never delete. First load = 1.3 GB download; subsequent = ~5 sec from disk.
- **Local embedding cache:** `models/*.npy` (gitignored). Embedder reuses if shape matches `(n_chunks, 1024)`.

## What's committed vs regenerable

**Committed (in git):**
- Source code: `*.ipynb`, `*.py` (retrieval, evaluate, explain, model_compare, ablation)
- `corpus/raw_pdfs/` — the 4 source PDFs
- `corpus/parsed/*.json` — parser output
- `corpus/chunks/all_chunks.jsonl` — chunker output (666 chunks — the shared source of truth)
- `corpus/manifest.json`, `corpus/SOURCES.md`, `corpus/download_sources.py`
- `eval/eval_set.json` — 18 labeled questions with verified gold chunk_ids
- `artifacts/day2/` — all evaluation outputs (JSON + CSV)
- `docs/` — decks + teammate guide + `DAY2_DELIVERABLES.md`

**Regenerable — gitignored:**
- `corpus/chunks/chunk_embeddings.npy` (~3 MB — regenerable via embedder)
- `corpus/chunks/chroma_db/` (~13 MB — regenerable via embedder)
- `models/*.npy` (~8 MB — regenerable via embedder/model_compare/ablation)
- `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`, Office lock files

## Measured Day-2 results (see docs/DAY2_DELIVERABLES.md for full tables)

| Metric | Value | Source |
|--------|------:|--------|
| Semantic Hit@5 | 0.867 | `artifacts/day2/evaluation.csv` |
| Hybrid Hit@5 (0.7/0.3 RRF) | 0.867 | same |
| BM25 Hit@5 | 0.733 | same |
| MedEmbed vs BGE-large (Hit@5) | 0.867 vs 0.733 | `artifacts/day2/model_compare.json` |
| Chunk ablation winner | 400/40 (Hit@5 0.933 page-level) | `artifacts/day2/chunk_ablation.json` |
| Top-k elbow | k=5 (Hit@k plateaus) | `artifacts/day2/topk_curve.csv` |
| OOS max cosine | 0.663 → **Day-4 refusal cutoff ≈ 0.70** | `artifacts/day2/out_of_scope.json` |

## Non-obvious behaviors (things I've hit)

1. **Character corruption in KDIGO PDFs.** `≥` renders as `$` (67 places) or `‡` (25 places). `pdf_parser.ipynb` normalizes them contextually (only when adjacent to a digit) so real dollar signs and legit footnote daggers are preserved.
2. **Ligatures.** `ﬁ`/`ﬂ`/`ﬀ`/`ﬃ`/`ﬄ` normalized to `fi`/`fl`/etc.
3. **NICE footer** is 3 separate lines (`Chronic kidney disease...`, `Page X`, `of 78`) — noise patterns handle each.
4. **KDIGO chapter titles** span two lines on some pages — hardcoded overrides in the parser config per doc.
5. **Chroma metadata must be flat scalars** (str/int/float/bool). Lists like `page_range` get split into `page_start`/`page_end`.
6. **Adding a document** = 1 config dict in `pdf_parser.ipynb` cell 3 + 1 line in `chunker.ipynb`'s `PARSED_FILES` + 1 line in `SKIP_PAGES`. `embedder.ipynb` needs zero changes.
7. **Chunker had a latent bug** (fixed): `chunks[-2] = chunks[-2] + "\n\n" + chunks.pop()` — evaluates the RHS after pop shrinks the list, so if `len(chunks) == 2` the LHS is out of range. Now pops first, then mutates the (now-last) item. Never fired at 550/55 but fired at 400/40 during ablation.
8. **The `models/` folder is on-disk convenience only** — everything can be re-embedded from scratch in ~20 min. Gitignored.

## Where the deliverable is

- **Day 1:** Searchable Vector DB with Metadata — DONE. Retrieval verified end-to-end.
- **Day 2:** Retrieval Optimization — DONE. All 6 deck items done, every claim traces to `artifacts/day2/*`. See `docs/DAY2_DELIVERABLES.md`.
- **Day 3:** Grounded generation + citation — DONE. `retrieval.py`'s `HybridRetriever.hybrid_search(query, k)` feeds `generation.KidneyRAGGenerator`. See `docs/DAY3_DELIVERABLES.md`.
- **Day 4:** Safety, verification, evaluation, web UI — DONE. `safety.py` adds claim extraction + verification + faithfulness + citation accuracy + 4-level uncertainty language. `evaluate_day4.py` writes `artifacts/day4/*`. `web/backend/app.py` + `web/frontend/*` are the demo UI (FastAPI + static). See `docs/DAY4_DELIVERABLES.md`.
- **Day 5 (next):** Deploy the web UI to Render/Railway, rehearse 3-question demo (strong / partial / refusal), rerun `evaluate_day4.py` for live faithfulness numbers, cut final deck.

## Day 4 additions — quick reference

| File | Role |
|---|---|
| `safety.py` | Claim extractor, LLM/similarity/NLI verifiers, faithfulness + citation-accuracy math, 4-level evidence-strength mapping |
| `evaluate_day4.py` | Full pipeline harness → `artifacts/day4/*.{csv,json,md}` |
| `test_safety.py` | 18 offline tests (no API needed) |
| `web/backend/app.py` | FastAPI wrapping retrieval + generation + safety. Endpoints: `/health`, `/api/sources`, `/api/ask`, `/docs` |
| `web/frontend/*` | Self-contained landing-page UI (dark theme, mobile-responsive, no external deps) |
| `artifacts/day4/` | evaluation_log.csv · summary.json · threshold_sweep.csv · adversarial_results.csv · responsible_ai_checklist.md |

Run the web app: `python -m uvicorn web.backend.app:app --port 8000` → open http://127.0.0.1:8000.
Rerun eval: `python evaluate_day4.py --skip-generation` (retrieval-only, deterministic) or `python evaluate_day4.py --verifier llm` (needs Gemini quota).

## Team roles

| # | Owner | Day 1–2 | Days 3–5 |
|---|-------|---------|----------|
| P1 | KDIGO owner | Parse + chunk KDIGO | Day-3 grounded generation |
| P2 | NICE owner | Parse + chunk NICE | Day-3 grounded generation |
| P3 | Index/retrieval | Embedder + Chroma + hybrid + eval | Day-4 retrieval tuning |
| P4 | Eval/safety | Eval set + questions + QC | Days 4 & 5 eval dashboard + demo |

## Things to NEVER do

- Never edit `all_chunks.jsonl` by hand — regenerate via chunker.
- Never commit `chroma_db/`, `*.npy`, `models/`, or `.env` (all gitignored).
- Never hardcode API keys. `generation.py` and `web/backend/app.py` load `.env` via `python-dotenv`. Add a key by editing `.env` (gitignored), never the code.
- Never delete `~/.cache/huggingface/` — re-triggers 1.3 GB MedEmbed download.
- Never use `--output foo.ipynb` on nbconvert — always `--inplace`.
- Never introduce backwards-compat shims — this is a 5-day hackathon.
- Never invent citations — every claim must trace to a real `chunk_id` returned by `HybridRetriever`.
- Never change the chunk contract (`chunk_id`, `document_name`, `source_url`, `page_number`, `page_range`, `section_title`, `token_count`, `text`). Everything downstream depends on it.
- Never swap embedding models without a full re-embed of both chunks AND queries with the same model.
