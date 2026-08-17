# Kidney-RAG — Project brief for Claude

> Read this first. Don't re-explore.

## What this is

Creativa × Orange AI Hackathon (Aug 16–20, 2026). A **clinical RAG** over official public CKD guidelines. Every clinical answer must trace to `document + section + page + chunk_id`. Core philosophy: **Fluent Answer ≠ Safe Answer.**

## Corpus (do not re-download)

| Doc | File | Pages | Role |
|-----|------|-------|------|
| KDIGO 2024 CKD Guideline (full) | `corpus/raw_pdfs/KDIGO_2024_CKD_Guideline_full.pdf` | 199 | Primary index — drug/therapy engine |
| NICE NG203 CKD | `corpus/raw_pdfs/NICE_NG203_CKD_assessment_and_management.pdf` | 78 | Primary index — diagnosis/pathway engine |
| KDIGO Summary of Recs | `corpus/raw_pdfs/KDIGO_2024_CKD_Guideline_Summary_Recommendations.pdf` | 21 | Held-out — answer key for eval |
| KDIGO Exec Summary | `corpus/raw_pdfs/KDIGO_2024_CKD_Guideline_Executive_Summary.pdf` | 18 | Team reference only |
| USPSTF CKD Screening | `corpus/raw_pdfs/USPSTF_CKD_Screening_Recommendation.pdf` | 6 | Safe-refusal demo (I-statement) |

## Pipeline

```
raw_pdfs/         pdf_parser.ipynb        chunker.ipynb         embedder.ipynb
 KDIGO   ──▶  parsed/kdigo_parsed.json  ──▶                 ┐
 NICE    ──▶  parsed/nice_parsed.json   ──▶  chunks/        ├─▶  chunk_embeddings.npy
                                              all_chunks    │      chroma_db/
                                                 .jsonl     ┘   (ckd_guidelines collection)
```

### Run each stage
```bash
python -m jupyter nbconvert --to notebook --execute pdf_parser.ipynb --inplace   # parse both PDFs
python -m jupyter nbconvert --to notebook --execute chunker.ipynb   --inplace   # 477 chunks -> JSONL
python -m jupyter nbconvert --to notebook --execute embedder.ipynb  --inplace   # embed + Chroma + test
```
Notebooks are self-contained. Each reads its input, writes its output, and is idempotent. Always use `--inplace` (never `--output foo.ipynb` — that creates duplicates).

## Chunk contract (frozen — do not change)

```json
{"chunk_id": "kdigo_p44_c01",
 "document_name": "KDIGO 2024 CKD Guideline",
 "source_url": "https://kdigo.org/...",
 "page_number": 44,
 "page_range": [44, 45],
 "section_title": "3.8 Mineralocorticoid receptor antagonists (MRA)",
 "token_count": 483,
 "text": "..."}
```

- `chunk_id` format: `<doc-short>_p<page>_c<index>` (unique, human-readable)
- Chroma stores `page_start` / `page_end` as separate ints instead of `page_range` list
- 477 chunks total (KDIGO 405 + NICE 72). Target ≤500, hard max 750 tokens. Median 493.
- KDIGO pages 34–53 (Summary of Recs) are **excluded** — they duplicate the body

## Stack decisions (locked)

- **Parser:** PyMuPDF (`fitz`) — pdfplumber glues words together on KDIGO's 2-column layout
- **Tokenizer:** `tiktoken` `cl100k_base`
- **Embeddings:** `abhinand/MedEmbed-large-v0.1` (1024-dim, bge-large fine-tuned on PubMed)
  - Local via `sentence-transformers`. **No API.** No rate limits.
  - Query prefix: `"Represent this medical question for retrieving relevant clinical guideline passages: "` (bge-style — query side only)
  - L2-normalized embeddings → cosine similarity = dot product
- **Vector DB:** ChromaDB persistent client at `corpus/chunks/chroma_db/`, cosine space, collection `ckd_guidelines`
- **Model cache:** default HF cache `~/.cache/huggingface/hub/`. Never delete. First load = download (~1.3 GB); subsequent loads = 5 sec from disk.

## What's committed vs regenerable

**Committed (in git):**
- All source notebooks (`*.ipynb`)
- `corpus/raw_pdfs/` (source PDFs)
- `corpus/parsed/*.json` (parser output)
- `corpus/chunks/all_chunks.jsonl` (chunker output)
- `corpus/manifest.json`, `corpus/SOURCES.md`, `corpus/download_sources.py`

**Regenerable — gitignored:**
- `corpus/chunks/chunk_embeddings.npy` (2 MB — regenerable via embedder)
- `corpus/chunks/chroma_db/` (~13 MB — regenerable via embedder)
- `.env`, `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`

## Non-obvious behaviors (things I've hit)

1. **Character corruption in KDIGO PDF.** `≥` renders as `$` (67 places) or `‡` (25 places). `pdf_parser.ipynb` normalizes them contextually (only when adjacent to a digit).
2. **Ligatures.** `ﬁ`/`ﬂ`/`ﬀ`/`ﬃ`/`ﬄ` normalized to `fi`/`fl`/etc.
3. **NICE footer** is 3 separate lines (`Chronic kidney disease...`, `Page X`, `of 78`) — noise patterns handle each.
4. **KDIGO chapter titles** span two lines on pages 90 and 131 — hardcoded overrides in the parser config.
5. **Chroma metadata must be flat scalars** (str/int/float/bool). Lists like `page_range` get split into `page_start`/`page_end`.
6. **Adding a document** = 1 config dict in `pdf_parser.ipynb` cell 3 + 1 entry in `chunker.ipynb`'s `PARSED_FILES`. Same pipeline runs.
7. **Deprecated: `google-generativeai` package.** Use `google-genai` if going back to Gemini. Current embedder is local (MedEmbed).

## Where the deliverable is

- **Day 1 target:** Searchable Vector DB with Metadata — **DONE.** Verified: query → Top-K → correct section/page.
- Best test hits: BP target (multi-source) sim=0.85, SGLT2i sim=0.81.
- One weak area to fix Day 2: pure-threshold questions ("albuminuria threshold") miss Table 3 on p22. **Fix:** add BM25 hybrid retrieval (deck lists this as required experiment).

## Team roles

| # | Owner | Day-1 job | Days 2–5 |
|---|-------|-----------|----------|
| P1 | KDIGO owner | Parse + chunk KDIGO | Day-3 grounded generation |
| P2 | NICE owner | Parse + chunk NICE | Day-3 grounded generation |
| P3 | Index/retrieval | Embedder + Chroma + baseline test | Days 2 & 4 retrieval tuning |
| P4 | Eval/safety | Scope + test questions + QC | Days 4 & 5 eval dashboard + demo |

## Things to NEVER do

- Never edit `all_chunks.jsonl` by hand — regenerate via chunker
- Never commit `chroma_db/`, `*.npy`, or `.env` (already in `.gitignore`)
- Never hardcode API keys — load from `.env` via `python-dotenv`
- Never delete `~/.cache/huggingface/` — re-triggers 1.3 GB download
- Never use `--output foo.ipynb` on nbconvert — always `--inplace`
- Never introduce backwards-compat shims — this is a 5-day hackathon
- Never invent citations — every rec must trace to a real chunk_id via retrieval
