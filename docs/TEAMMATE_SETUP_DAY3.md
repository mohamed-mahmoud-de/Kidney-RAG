# Kidney-RAG — Day 3 Setup Guide

## Quick start (5 minutes)

### 1. Pull the latest code

```bash
git pull origin main
```

### 2. Install new dependencies

```bash
pip install -r requirements.txt
```

New packages added for Day 3: `huggingface_hub`, `google-genai`, `python-dotenv`.

### 3. Create your `.env` file

Create a file named `.env` in the project root (same folder as `generation.py`).
This file is **gitignored** — never commit it.

> **Day 4 update:** the old `KIDNEY_RAG_BACKEND=gemini/huggingface/anthropic`
> selector is gone. Kidney-RAG now uses `llm_pool.py` — you add **every key
> you have** and the pool round-robins them, fails over between providers
> when one hits a rate limit, and never asks you to pick one. See
> `.env.example` for the full template.

```env
# Gemini — add up to N keys (auto-detected as GOOGLE_API_KEY, _2, _3, ...)
GOOGLE_API_KEY=your-first-gemini-key
GOOGLE_API_KEY_2=your-second-gemini-key

# HuggingFace — same pattern
HF_TOKEN=hf_your_first_token
HF_TOKEN_2=hf_your_second_token

# Anthropic — optional third tier
# ANTHROPIC_API_KEY=sk-ant-your_key_here

# Optional: change tier order (default gemini,huggingface,anthropic)
# LLM_POOL_ORDER=gemini,huggingface
```

**How to get a Google API key (free):**
1. Go to [aistudio.google.com](https://aistudio.google.com/)
2. Sign in with your Google account
3. Click "Get API Key" → "Create API Key"
4. Copy and paste it into your `.env`

**How to get an HF token (free):**
1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a new token with "Read" + "Make calls to Inference Providers" permissions
3. Copy and paste into your `.env`

### 4. Verify the index exists

The retriever needs the Chroma vector DB and chunk embeddings.
If you already ran the Day 1-2 notebooks, these exist. Check:

```bash
python -c "from retrieval import HybridRetriever; r = HybridRetriever(); print(f'OK: {len(r.chunks)} chunks, {r.collection.count()} vectors')"
```

Expected output: `OK: 666 chunks, 666 vectors`

If it fails, rebuild the index (~15-20 min first run):

```bash
python -m jupyter nbconvert --to notebook --execute embedder.ipynb --inplace
```

### 5. Run the offline tests

```bash
python test_generation.py
```

Expected: `18/18 passed`. These don't need any API key.

### 6. Run the full pipeline notebook

Open `day3_pipeline.ipynb` in Jupyter or VS Code and run all cells.
Or from the command line:

```bash
python -m jupyter nbconvert --to notebook --execute day3_pipeline.ipynb --inplace
```

This runs the full Day 3 demo: Cases A/B/C, USPSTF screening, and adversarial stress tests.

---

## What was built on Day 3

| File | What it does |
|---|---|
| `generation.py` | The generation engine: retrieval gate + prompt assembly + LLM call + format validation |
| `kidney_rag_system_prompt.md` | System prompt that constrains the LLM to only answer from retrieved chunks |
| `test_generation.py` | 18 offline unit tests (no API needed) |
| `day3_pipeline.ipynb` | Interactive demo notebook with saved outputs |
| `eval/day5_refusal_demo.json` | Rehearsed refusal case for the Day-5 live demo |
| `docs/DAY3_DELIVERABLES.md` | Full deliverables summary with metrics |

## How it works

You type a clinical question → the system:

1. **Retrieves** top-5 chunks from the CKD guidelines vector DB (semantic + BM25 hybrid)
2. **Gates** on quality: if the top hit's cosine similarity < 0.70 → instant refusal, no LLM call
3. **Assembles** a grounded prompt: system rules + retrieved chunk text + pre-built citations
4. **Generates** an answer via the LLM (constrained to only use the retrieved text)
5. **Validates** the output format (must have Recommendation + Excerpt + Citation)
6. **Appends** a clinical disclaimer

### Quick test from Python

```python
from retrieval import HybridRetriever
from generation import KidneyRAGGenerator

retriever = HybridRetriever()
generator = KidneyRAGGenerator(retriever)

# In-scope question → cited answer
result = generator.answer("What daily protein intake is recommended for CKD G3-G5?")
print(result.text)

# Out-of-scope question → clean refusal
result = generator.answer("What antibiotic treats pneumonia?")
print(result.text)  # "I can't answer this from the available sources..."
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'google.genai'` | Run `pip install google-genai` |
| `ModuleNotFoundError: No module named 'huggingface_hub'` | Run `pip install huggingface_hub` |
| `ValueError: Set HF_TOKEN env var` | Add `HF_TOKEN=hf_xxx` to your `.env` file |
| `402 Payment Required` (HuggingFace) | Monthly credits depleted. The pool will auto-fail-over to another tier if you have Gemini/Anthropic keys set; otherwise upgrade to HF PRO ($9/mo, 20x credits) or wait for reset. |
| `429 RESOURCE_EXHAUSTED` (Gemini) | Rate limited. The pool cools the key down and rotates to another. Free tier: 5 rpm, 20/day per key — add a second key (`GOOGLE_API_KEY_2`) to double the ceiling. |
| `LLMPoolExhausted` at runtime | Every key across every tier is in cooldown. Check `/health` in the web app to see per-key state and next-available time. |
| `404 model not found` (Gemini) | Update `google-genai`: `pip install --upgrade google-genai` |
| `Chroma has X vectors but chunks file has Y rows` | Re-run `embedder.ipynb` to rebuild the index |
| Retriever takes ~30 sec to load | Normal on first run — MedEmbed loads from HF cache (~1.3 GB) |

## Project structure (after Day 3)

```
Kidney-RAG/
├── corpus/
│   ├── raw_pdfs/          ← 4 source guideline PDFs
│   ├── parsed/            ← PDF parser output (JSON)
│   ├── chunks/
│   │   ├── all_chunks.jsonl   ← 666 chunks (source of truth)
│   │   └── chroma_db/        ← vector index (gitignored, regenerable)
│   └── manifest.json
├── eval/
│   ├── eval_set.json          ← 18 labeled questions
│   └── day5_refusal_demo.json ← rehearsed refusal case
├── artifacts/day2/            ← Day 2 evaluation outputs
├── docs/
│   ├── DAY2_DELIVERABLES.md
│   ├── DAY3_DELIVERABLES.md
│   └── TEAMMATE_SETUP_DAY3.md ← this file
├── pdf_parser.ipynb           ← Day 1: PDF → JSON
├── chunker.ipynb              ← Day 1: JSON → chunks
├── embedder.ipynb             ← Day 1: chunks → vectors
├── retrieval.py               ← Day 2: hybrid retriever
├── evaluate.py                ← Day 2: eval metrics
├── generation.py              ← Day 3: grounded generation
├── kidney_rag_system_prompt.md ← Day 3: system prompt
├── test_generation.py         ← Day 3: offline tests
├── day3_pipeline.ipynb        ← Day 3: full demo notebook
├── .env                       ← YOUR API keys (gitignored)
├── .gitignore
├── requirements.txt
└── CLAUDE.md
```
