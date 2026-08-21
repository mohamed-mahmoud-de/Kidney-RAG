---
title: Kidney-RAG
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Clinical RAG over CKD guidelines with citations and safety scoring.
---

# Kidney-RAG

**Kidney-RAG** is a research prototype for retrieving evidence from authoritative chronic kidney disease (CKD) guidelines. It converts public guideline PDFs into citation-ready passages, indexes them with a local medical embedding model and ChromaDB, and provides semantic, lexical, and hybrid retrieval experiments.

The project is designed around one safety principle:

> **A fluent answer is not necessarily a safe answer.** Every retrieved recommendation should remain traceable to its source document, section, page range, and `chunk_id`.

This repository ships the **full four-day hackathon build**: retrieval foundation (Days 1–2), grounded generation with a citation-shaped answer schema and a calibrated refusal gate (Day 3), and a safety layer with claim-level faithfulness scoring, a multi-key LLM pool, and a deployable web UI (Day 4). It is a research prototype and does not replace professional medical judgment.

For a plain-English walkthrough of what shipped each day, see [`docs/KIDNEY_RAG_TEAMMATE_GUIDE.pdf`](docs/KIDNEY_RAG_TEAMMATE_GUIDE.pdf).

## What the project does

The pipeline performs four main stages:

1. **Parse** official CKD guideline PDFs with PyMuPDF, including cleanup for PDF layout artifacts, ligatures, footers, and corrupted threshold characters.
2. **Chunk** parsed text at section-aware boundaries while preserving citation metadata and avoiding duplicate or non-clinical material that can distort retrieval.
3. **Embed and index** the chunks locally with `abhinand/MedEmbed-large-v0.1` and a persistent ChromaDB collection using cosine similarity.
4. **Retrieve** relevant passages with semantic search, BM25 keyword search, or weighted Reciprocal Rank Fusion (RRF), which combines semantic meaning with exact clinical terms and thresholds.

The intended downstream flow is:

```text
User question
     │
     ▼
Semantic retrieval ──────┐
                         ├── weighted RRF ──► top-k cited guideline chunks
BM25 lexical retrieval ─┘
```

## Current corpus

The committed chunk artifact contains **666 retrieval-ready chunks** from four guideline sources. The repository also includes the source PDFs and parsed JSON artifacts used to reproduce the pipeline. Source provenance and download status are documented in [`corpus/SOURCES.md`](corpus/SOURCES.md) and [`corpus/manifest.json`](corpus/manifest.json).

| Source | Publisher | Role in the corpus | Indexed chunks |
|---|---|---|---:|
| [KDIGO 2024 CKD Guideline][1] | KDIGO | CKD evaluation and management | 340 |
| [KDIGO 2022 Diabetes Management in CKD][2] | KDIGO | Diabetes-specific CKD management | 241 |
| [NICE NG203: CKD assessment and management][3] | NICE | Assessment, classification, and management pathways | 72 |
| [USPSTF: Screening for CKD][4] | USPSTF | Screening evidence and safe-refusal cases | 13 |
| **Total** |  |  | **666** |

The chunker uses a target size of **550 `cl100k_base` tokens**, **55-token overlap** when a section must be split, and a hard maximum of **750 tokens**. Recommendations and practice points are kept intact where possible, and each chunk remains associated with a section title.

To reduce duplicate retrieval and noise, the pipeline excludes duplicated summary page ranges and filters non-clinical sections such as research recommendations, acknowledgements, administrative material, and author rosters. The KDIGO foreword is intentionally retained because it contains canonical CKD, GFR, and albuminuria staging tables.

## Repository structure

```text
Kidney-RAG/
├── corpus/
│   ├── raw_pdfs/                 # Public guideline PDFs
│   ├── parsed/                   # Page- and section-aware parser outputs
│   ├── chunks/
│   │   └── all_chunks.jsonl      # Committed 666-chunk retrieval corpus
│   ├── SOURCES.md                # Source list, provenance, and URLs
│   ├── manifest.json             # Download status, hashes, and metadata
│   └── download_sources.py       # Optional source downloader/checker
├── eval/
│   └── eval_set.json             # 18-question labeled retrieval set
├── docs/                         # Team onboarding and project materials
├── pdf_parser.ipynb              # PDF → parsed JSON
├── chunker.ipynb                 # Parsed JSON → citation-ready JSONL
├── embedder.ipynb                # JSONL → embeddings + ChromaDB
├── retriever.ipynb               # Day 2 hybrid retrieval sandbox (superseded by retrieval.py)
├── retrieval.py                  # HybridRetriever (production entry point)
├── evaluate.py                   # Day 2 retrieval eval — writes artifacts/day2/*
├── generation.py                 # Day 3 grounded generator + quality gate + refusal
├── safety.py                     # Day 4 claim extractor + faithfulness + citation accuracy
├── llm_pool.py                   # Day 4 multi-key LLM pool with RR + failover
├── evaluate_day4.py              # Day 4 full-pipeline eval — writes artifacts/day4/*
├── test_generation.py            # 18 offline tests
├── test_safety.py                # 18 offline tests
├── test_llm_pool.py              # 18 offline tests
├── web/backend/app.py            # FastAPI wrapping the pipeline
├── web/frontend/                 # Landing-page UI (self-contained)
├── artifacts/day2/               # Day 2 measurement outputs
├── artifacts/day4/               # Day 4 measurement outputs + responsible AI checklist
├── requirements.txt              # Pinned Python dependencies
├── .env.example                  # Multi-key template for the LLM pool
├── CLAUDE.md                     # Internal project brief and implementation contracts
└── README.md
```

The embedding cache and ChromaDB directory are intentionally regenerable and gitignored:

```text
corpus/chunks/chunk_embeddings.npy
corpus/chunks/chroma_db/
```

## Requirements

The project is intended for **Python 3.11 or newer**. The dependencies are pinned in [`requirements.txt`](requirements.txt). A local CPU environment is sufficient, although embedding the corpus is substantially faster with suitable hardware.

The first embedding run downloads the MedEmbed model to the Hugging Face cache. Subsequent runs reuse the local model cache and, when available, the cached `chunk_embeddings.npy` file.

## Installation

```bash
git clone https://github.com/mohamed-mahmoud-de/Kidney-RAG.git
cd Kidney-RAG

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Start Jupyter when you want to inspect or execute the notebooks:

```bash
jupyter lab
```

No API key is required for the current embedding and retrieval pipeline. The local MedEmbed model is used instead of an API-based embedding service. The `.env.example` file is retained for future API-based generation experiments; never commit real credentials.

## Reproduce the pipeline

Run the notebooks in this order from the repository root. Each notebook reads the previous stage's output and writes its own output.

```bash
python -m jupyter nbconvert --to notebook --execute pdf_parser.ipynb --inplace
python -m jupyter nbconvert --to notebook --execute chunker.ipynb --inplace
python -m jupyter nbconvert --to notebook --execute embedder.ipynb --inplace
```

The stages produce the following artifacts:

| Stage | Input | Output |
|---|---|---|
| Parsing | PDFs in `corpus/raw_pdfs/` | JSON files in `corpus/parsed/` |
| Chunking | Parsed JSON files | `corpus/chunks/all_chunks.jsonl` |
| Embedding and indexing | `all_chunks.jsonl` | `chunk_embeddings.npy` and `chroma_db/` |
| Retrieval experiments | ChromaDB plus `all_chunks.jsonl` | Notebook output only |

The notebooks are designed to be rerunnable. Use `--inplace` so execution updates the existing notebook rather than creating duplicate notebook files.

## Retrieval modes

### Semantic retrieval

[`embedder.ipynb`](embedder.ipynb) uses [`abhinand/MedEmbed-large-v0.1`][5], a 1024-dimensional medical embedding model loaded through `sentence-transformers`. Embeddings are L2-normalized and stored in a persistent ChromaDB collection named `ckd_guidelines` under cosine distance.

Queries use the following instruction prefix:

```text
Represent this medical question for retrieving relevant clinical guideline passages:
```

### Hybrid retrieval

[`retriever.ipynb`](retriever.ipynb) combines semantic retrieval with BM25 lexical retrieval using weighted Reciprocal Rank Fusion. Rank fusion is used instead of adding raw scores because BM25 and cosine similarity have different numeric scales.

The current recommended defaults are:

| Parameter | Default | Purpose |
|---|---:|---|
| Semantic weight | `0.7` | Prioritizes paraphrase and meaning matching |
| Lexical weight | `0.3` | Helps match exact drug names, numbers, and thresholds |
| RRF constant | `60` | Dampens the effect of rank position |
| Candidate pool | `50` | Candidates contributed by each retriever before fusion |

The notebook exposes `cosine_search`, `bm25_search`, `weighted_rrf`, and `hybrid_search(query, k)`. The hybrid retriever is a read-only consumer of the existing index and is intended to be the retrieval entry point for a later grounded generation stage.

## Citation metadata

Every chunk carries the metadata needed for a grounded response. A representative record has this shape:

```json
{
  "chunk_id": "kdigo_p44_c01",
  "document_name": "KDIGO 2024 CKD Guideline",
  "source_url": "https://kdigo.org/...",
  "page_number": 44,
  "page_range": [44, 45],
  "section_title": "3.8 Mineralocorticoid receptor antagonists (MRA)",
  "token_count": 483,
  "text": "..."
}
```

When metadata is stored in ChromaDB, the page range is represented as the flat scalar fields `page_start` and `page_end`, because ChromaDB metadata must use scalar values. Do not hand-edit `all_chunks.jsonl`; regenerate it with the chunker when the parsing or chunking logic changes.

## Evaluation set

[`eval/eval_set.json`](eval/eval_set.json) contains **18 labeled questions** covering direct factual retrieval, multi-source questions, edge cases, screening evidence, and out-of-scope requests. Each in-scope question is paired with verified gold `chunk_id` values.

The dataset defines three basic retrieval metrics:

| Metric | Definition |
|---|---|
| Precision@k | Number of gold chunks in the top `k`, divided by `k` |
| Recall@k | Number of gold chunks in the top `k`, divided by the number of gold chunks |
| Hit@k | `1` when any gold chunk appears in the top `k`; otherwise `0` |

The evaluation set also tests behavior beyond ranking quality. Screening questions should surface the USPSTF **insufficient-evidence** conclusion rather than force a yes/no recommendation, while out-of-scope questions should be refused instead of being answered with an unrelated CKD passage.

## Grounded generation (Day 3)

[`generation.py`](generation.py) turns the top-k retrieval into a strictly grounded answer. Every response comes back in a fixed **Recommendation / Excerpt / Citation** shape, with the citation built from real chunk metadata (`chunk_id`, `document_name`, `section_title`, `page_number` / `page_range`, `source_url`) so it is programmatically verifiable.

A **retrieval-quality gate** at cosine ≥ 0.70 (calibrated on `eval/eval_set.json` — 0.10 gap between in-scope and out-of-scope questions) refuses before the LLM is ever called, so out-of-scope queries never risk hallucination. See [`docs/DAY3_DELIVERABLES.md`](docs/DAY3_DELIVERABLES.md) for the full flow and refusal conditions.

## Safety layer + evaluation (Day 4)

[`safety.py`](safety.py) scores every generated answer:

- **Claim extraction** — splits the Recommendation into atomic factual claims, attaches cited chunk_ids.
- **Verification** — LLM verifier (primary), embedding similarity (offline fallback), or NLI cross-encoder (opt-in).
- **Faithfulness** = supported claims / total claims.
- **Citation accuracy** = correct cited chunks / total cited chunks (existence + support).
- **4-level evidence-strength labels** (strong / partial / weak / insufficient) derived from top-hit cosine, driving the UI badge and the wrapper phrase in the answer.

[`llm_pool.py`](llm_pool.py) holds a rotating pool of provider keys. Round-robin within a tier (double your effective Gemini quota with two keys), failover across tiers when a whole tier is in cooldown (Gemini → HuggingFace → Anthropic). Role-based tier routing: generation prefers Gemini, verification prefers HuggingFace to preserve Gemini quota.

[`evaluate_day4.py`](evaluate_day4.py) is the full-pipeline harness that writes `artifacts/day4/{evaluation_log.csv, threshold_sweep.csv, adversarial_results.csv, summary.json, responsible_ai_checklist.md}`. See [`docs/DAY4_DELIVERABLES.md`](docs/DAY4_DELIVERABLES.md) for the metric tables and threshold justification.

## Web app

[`web/backend/app.py`](web/backend/app.py) is a FastAPI process that wraps retriever + generator + safety layer. Endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /` | Static landing page |
| `POST /api/ask` | Retrieval → gate → generation → safety scoring |
| `GET /health` | Retriever + generator + live per-key pool status |
| `GET /api/sources` | Catalog of indexed guidelines |
| `GET /api/logs` | Download the full site-query CSV audit log |
| `GET /api/stats` | Aggregate: refusals, avg faithfulness, latency, evidence-strength breakdown |
| `GET /docs` | Auto-generated OpenAPI browser |

The frontend at [`web/frontend/`](web/frontend) is a self-contained HTML/CSS/JS landing page — dark medical-tech aesthetic, mobile-responsive, no external dependencies. Every `/api/ask` call is appended to `logs/site_queries.csv` for post-hoc analysis.

Run locally:

```bash
python -m uvicorn web.backend.app:app --port 8000
# open http://127.0.0.1:8000
```

## Tests

54 offline tests across three suites, all runnable without API keys or network:

```bash
python test_generation.py         # 18 tests — Day 3 gate + citation + format
python test_safety.py             # 18 tests — Day 4 extractor + scoring math
python -m pytest test_llm_pool.py # 18 tests — pool RR, cooldown, failover
```

## Safety and scope

This repository indexes guideline text for research and engineering experimentation. It is **not medical advice**, does not diagnose or treat patients, and should not be used as a substitute for a qualified clinician, institutional protocol, or the current official guideline.

A future answer-generation layer should preserve the following boundaries:

- Cite the retrieved source document, section, page, and `chunk_id` for each clinical claim.
- Distinguish recommendations from practice points, evidence statements, and insufficient-evidence conclusions.
- Refuse questions outside the indexed CKD scope rather than hallucinating an answer.
- Surface conflicts between sources, such as differing blood-pressure target definitions, instead of silently merging them.
- Verify the current version and applicability of a guideline before using it in clinical practice.

The corpus is based on public documents, but the presence of a passage in the index does not guarantee that it is current, complete, or appropriate for a particular patient.

## Development guidelines

Keep the chunk schema stable because downstream retrieval and evaluation depend on `chunk_id`, document identity, page metadata, section title, and source URL. When adding a source document, update the parser configuration, chunker input list, source manifest, and evaluation coverage together.

Generated artifacts such as ChromaDB, NumPy embedding caches, virtual environments, and `.env` files should remain uncommitted. The project brief in [`CLAUDE.md`](CLAUDE.md) records additional implementation contracts and known PDF parsing edge cases.

## License and attribution

No license file is currently included in the repository. Before redistributing the code or bundled PDFs, review the licensing and reuse terms of the project and each source publisher. The guideline documents remain the property of their respective publishers.

## References

[1]: https://kdigo.org/wp-content/uploads/2024/03/KDIGO-2024-CKD-Guideline.pdf "KDIGO 2024 Clinical Practice Guideline for the Evaluation and Management of CKD"
[2]: https://kdigo.org/wp-content/uploads/2022/10/KDIGO-2022-Clinical-Practice-Guideline-for-Diabetes-Management-in-CKD.pdf "KDIGO 2022 Clinical Practice Guideline for Diabetes Management in CKD"
[3]: https://www.nice.org.uk/guidance/ng203/resources/chronic-kidney-disease-assessment-and-management-pdf-66143713055173 "NICE NG203: Chronic kidney disease: assessment and management"
[4]: https://www.uspreventiveservicestaskforce.org/home/getfilebytoken/ZRz9nTrjKkRtNTe6hgPze- "USPSTF Recommendation Statement: Screening for Chronic Kidney Disease"
[5]: https://huggingface.co/abhinand/MedEmbed-large-v0.1 "MedEmbed-large-v0.1 model card"
