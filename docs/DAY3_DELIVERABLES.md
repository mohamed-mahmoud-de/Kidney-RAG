# Kidney-RAG — Day 3 Deliverables

## Scope

Day 3 builds the grounded generation layer on top of Day 2's retriever.
Every clinical answer is now structured (Recommendation / Excerpt / Citation),
gate-checked on retrieval quality, and refuses cleanly when evidence is insufficient.

## What's new

| File | Purpose |
|---|---|
| `generation.py` | `KidneyRAGGenerator` — gate + prompt assembly + LLM call + format validation. Supports 3 backends: HuggingFace, Gemini, Anthropic. |
| `kidney_rag_system_prompt.md` | Grounding prompt: role, no-outside-knowledge rule, required rec/excerpt/citation structure, 6 refusal conditions, citation format. |
| `test_generation.py` | 18 offline tests (gate logic, citation formatting, output validation, refusal shape). No API calls. |
| `day3_pipeline.ipynb` | Full interactive demo notebook — Cases A/B/C, USPSTF, adversarial stress tests, gate calibration table. Outputs saved. |
| `eval/day5_refusal_demo.json` | Rehearsed Day-5 refusal case (pneumonia antibiotic → gate-level refusal). |

## Pipeline flow

```
User question
     │
     ▼
HybridRetriever.hybrid_search(query, k=5)
  ├─ Semantic: MedEmbed cosine (0.7 weight)
  └─ Lexical:  BM25 (0.3 weight)
     │
     ▼
Quality Gate — cosine_sim ≥ 0.70?
  ├─ FAIL → instant refusal (no LLM call)
  └─ PASS → confidence = high (≥0.80) / medium (0.70–0.80)
     │
     ▼
Grounded prompt (system prompt + retrieved chunks + pre-built citations)
     │
     ▼
LLM call (Gemini 3.6 Flash / Qwen 7B / Claude)
     │
     ▼
Format validation → Clinical disclaimer → Answer
```

## Gate calibration (from eval/eval_set.json, 18 questions)

| Group | cosine_sim range | Gate result |
|---|---|---|
| In-scope (15 questions) | 0.7377 – 0.8937 | All PASS |
| Out-of-scope (3 questions) | 0.6185 – 0.6330 | All REFUSE |
| **Separation gap** | **0.1047** | Threshold = **0.70** |

## Day 3 definition of done

| Requirement | Status | Evidence |
|---|---|---|
| Grounding prompt survives adversarial tests | ✓ | 5/5 adversarial probes refused (`day3_pipeline.ipynb`) |
| Every answer structured as rec/excerpt/citation | ✓ | `validate_output_format()` + prompt rules |
| Citation = document + section + page every time | ✓ | Fixed schema in `format_citation()`, page ranges supported |
| Refusal logic triggers correctly | ✓ | Gate-level (cosine < 0.70) + model-level (6 conditions in prompt) |
| Rehearsed refusal case for Day 5 | ✓ | `eval/day5_refusal_demo.json` (pneumonia antibiotic) |
| Full pipeline end-to-end | ✓ | Cases A, B, C, USPSTF — all in `day3_pipeline.ipynb` |

## LLM backend config

Set in `.env` (gitignored, never committed):

| Variable | Options | Default |
|---|---|---|
| `KIDNEY_RAG_BACKEND` | `gemini` / `huggingface` / `anthropic` | `huggingface` |
| `GOOGLE_API_KEY` | Google AI Studio key | (required for gemini) |
| `HF_TOKEN` | HuggingFace access token | (required for huggingface) |
| `ANTHROPIC_API_KEY` | Anthropic API key | (required for anthropic) |

## Measured results

- **Gate accuracy**: 18/18 correct decisions on eval set (perfect separation)
- **Format compliance**: All in-scope answers validated with `validate_output_format()` = True
- **Adversarial resistance**: 5/5 probes (outside knowledge, bypass attempt, fake emergency, dosage prescription, personal diagnosis) → correctly refused
- **Rehearsed refusal**: Instant, deterministic, no LLM call (gate-level)
