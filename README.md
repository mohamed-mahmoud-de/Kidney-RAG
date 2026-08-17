# Kidney-RAG

**Kidney-RAG** is a research prototype for retrieving evidence from authoritative chronic kidney disease (CKD) guidelines. It converts public guideline PDFs into citation-ready passages, indexes them with a local medical embedding model and ChromaDB, and provides semantic, lexical, and hybrid retrieval experiments.

The project is designed around one safety principle:

> **A fluent answer is not necessarily a safe answer.** Every retrieved recommendation should remain traceable to its source document, section, page range, and `chunk_id`.

This repository currently focuses on the **retrieval foundation** of a clinical RAG system. It does not provide a production clinical decision-support application or replace professional medical judgment.

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
├── retriever.ipynb               # Semantic + BM25 hybrid retrieval sandbox
├── requirements.txt              # Pinned Python dependencies
├── .env.example                  # Optional API-key template for future work
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
