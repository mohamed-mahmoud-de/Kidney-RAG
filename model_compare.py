"""Embedding-model head-to-head on the labeled eval set (deck requirement).

Compares two SentenceTransformer models on the SAME 666 chunks + SAME 15
in-scope questions:
  * MedEmbed-large-v0.1 (medical fine-tune of bge-large)  — 1024-dim
  * BAAI/bge-large-en-v1.5 (general-purpose baseline)     — 1024-dim

For each model:
  1. Embed all chunks (cached to models/embeddings_<slug>.npy after first run)
  2. Embed each query and compute cosine similarity to every chunk (in-memory)
  3. Score Precision@5, Recall@5, Hit@5 against eval/eval_set.json gold

Cosine similarity is computed manually (dot product of L2-normalized vectors)
so we do NOT need to rebuild ChromaDB per model.

Usage:
    python model_compare.py
    python model_compare.py --k 3
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")


MODELS = [
    {
        "slug":  "medembed_large",
        "name":  "abhinand/MedEmbed-large-v0.1",
        "label": "MedEmbed-large (medical fine-tune of bge-large)",
        # bge-family models use a query-side instruction prefix
        "query_prefix": "Represent this medical question for retrieving relevant clinical guideline passages: ",
    },
    {
        "slug":  "bge_large_en_v1_5",
        "name":  "BAAI/bge-large-en-v1.5",
        "label": "BGE-large-en-v1.5 (general English baseline)",
        # bge-large uses this canonical prefix
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
]

CHUNKS_PATH = "corpus/chunks/all_chunks.jsonl"
EVAL_PATH   = "eval/eval_set.json"
CACHE_DIR   = "models"


def load_chunks(path=CHUNKS_PATH):
    chunks = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    return chunks


def load_cases(path=EVAL_PATH):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = payload.get("questions", payload) if isinstance(payload, dict) else payload
    return [{
        "id": q["id"],
        "category": q["category"],
        "query": q.get("query", q.get("question")),
        "expected_chunk_ids": q.get("expected_chunk_ids", q.get("gold_chunk_ids", [])),
    } for q in raw]


def embed_or_load(model_spec, texts):
    """Cache embeddings per model so re-runs skip the 15-20 min work."""
    Path(CACHE_DIR).mkdir(exist_ok=True)
    cache_path = Path(CACHE_DIR) / f"embeddings_{model_spec['slug']}.npy"

    if cache_path.exists():
        vecs = np.load(cache_path)
        if vecs.shape[0] == len(texts):
            print(f"  [cache] reused {cache_path}  shape={vecs.shape}")
            return vecs, cache_path
        print(f"  [cache stale] shape mismatch — re-embedding")

    from sentence_transformers import SentenceTransformer
    print(f"  [load] {model_spec['name']} ...")
    t0 = time.time()
    m = SentenceTransformer(model_spec["name"])
    print(f"  [load] done in {time.time()-t0:.1f}s")
    dim = m.get_sentence_embedding_dimension()

    print(f"  [embed] {len(texts)} chunks, dim={dim} — this can take ~15-20 min on CPU")
    t0 = time.time()
    vecs = m.encode(texts, batch_size=32, show_progress_bar=True,
                    normalize_embeddings=True, convert_to_numpy=True)
    print(f"  [embed] done in {time.time()-t0:.1f}s  shape={vecs.shape}")
    np.save(cache_path, vecs)
    print(f"  [save] {cache_path}")
    return vecs, cache_path


def encode_queries(model_spec, queries):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_spec["name"])
    prefix = model_spec["query_prefix"]
    return m.encode([prefix + q for q in queries],
                    normalize_embeddings=True, convert_to_numpy=True)


def topk_ids(query_vec, doc_vecs, chunk_ids, k):
    # L2-normalized -> cosine = dot product
    sims = doc_vecs @ query_vec
    idx = np.argsort(sims)[::-1][:k]
    return [chunk_ids[i] for i in idx], [float(sims[i]) for i in idx]


def precision_at_k(retrieved, expected, k):
    if not expected: return 0.0
    return sum(cid in set(expected) for cid in retrieved[:k]) / k


def recall_at_k(retrieved, expected, k):
    if not expected: return 0.0
    return sum(cid in retrieved[:k] for cid in expected) / len(expected)


def hit_at_k(retrieved, expected, k):
    if not expected: return 0
    return int(any(cid in retrieved[:k] for cid in expected))


def evaluate_model(model_spec, chunks, cases, k):
    print(f"\n=== {model_spec['label']} ===")
    doc_vecs, _ = embed_or_load(model_spec, [c["text"] for c in chunks])
    chunk_ids = [c["chunk_id"] for c in chunks]

    in_scope = [c for c in cases if c["category"] != "out_of_scope"]
    oos      = [c for c in cases if c["category"] == "out_of_scope"]

    q_vecs = encode_queries(model_spec, [c["query"] for c in in_scope])
    p_scores, r_scores, h_scores = [], [], []
    per_row = []
    for c, qv in zip(in_scope, q_vecs):
        got_ids, sims = topk_ids(qv, doc_vecs, chunk_ids, k)
        p = precision_at_k(got_ids, c["expected_chunk_ids"], k)
        r = recall_at_k(got_ids, c["expected_chunk_ids"], k)
        h = hit_at_k(got_ids, c["expected_chunk_ids"], k)
        p_scores.append(p); r_scores.append(r); h_scores.append(h)
        per_row.append({"id": c["id"], "category": c["category"], "p": round(p,3),
                        "r": round(r,3), "hit": h, "top1_sim": round(sims[0], 3)})

    # OOS: top-1 cosine per question (refusal-threshold data)
    oos_report = []
    if oos:
        oos_q_vecs = encode_queries(model_spec, [c["query"] for c in oos])
        for c, qv in zip(oos, oos_q_vecs):
            _, sims = topk_ids(qv, doc_vecs, chunk_ids, 1)
            oos_report.append({"id": c["id"], "top1_sim": round(sims[0], 3)})

    return {
        "model": model_spec["name"],
        "label": model_spec["label"],
        "k": k,
        "n_scored": len(in_scope),
        "precision_at_k": round(np.mean(p_scores), 4),
        "recall_at_k":    round(np.mean(r_scores), 4),
        "hit_at_k":       round(np.mean(h_scores), 4),
        "oos_max_top1_sim": round(max(o["top1_sim"] for o in oos_report), 4) if oos_report else None,
        "per_row": per_row,
        "oos": oos_report,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--output-dir", default="artifacts/day2")
    args = ap.parse_args()

    chunks = load_chunks()
    cases  = load_cases()
    print(f"Loaded {len(chunks)} chunks and {len(cases)} eval cases (k={args.k})")

    results = [evaluate_model(spec, chunks, cases, args.k) for spec in MODELS]

    # Summary table
    print("\n" + "=" * 90)
    print(f"HEAD-TO-HEAD @ k={args.k}   (n_scored={results[0]['n_scored']} in-scope questions)")
    print("=" * 90)
    print(f"{'Model':<55} {'P@k':>8} {'R@k':>8} {'Hit@k':>8} {'OOS-top1':>10}")
    for r in results:
        print(f"{r['label']:<55} {r['precision_at_k']:>8.4f} {r['recall_at_k']:>8.4f} {r['hit_at_k']:>8.4f} {str(r['oos_max_top1_sim']):>10}")

    # Winner
    winner = max(results, key=lambda r: (r["hit_at_k"], r["precision_at_k"]))
    print(f"\nWinner (by Hit@k then P@k): {winner['label']}")

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "model_compare.json").write_text(
        json.dumps({"k": args.k, "results": results, "winner": winner["model"]},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWritten: {out / 'model_compare.json'}")


if __name__ == "__main__":
    main()
