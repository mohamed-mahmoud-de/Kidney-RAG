"""Run the labeled retrieval evaluation against eval/eval_set.json.

Computes for each in-scope question:
  * Precision@k and Recall@k for semantic / BM25 / hybrid
  * Hit@k (any gold chunk retrieved)
  * Latency per retrieval method

Also reports out-of-scope max-similarity (the max cosine seen for any question
whose gold_chunk_ids is empty) — this is the empirical floor for the Day-4
refusal threshold: any real query scoring below it lacks any supporting
guideline chunk in our corpus, so the system should refuse.

Usage:
    python evaluate.py                               # defaults: k=5, hybrid 0.7/0.3
    python evaluate.py --k 3
    python evaluate.py --sweep-k                     # writes P@1..P@10 curve
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from retrieval import HybridRetriever


def load_cases(path: str) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = payload.get("questions", payload) if isinstance(payload, dict) else payload
    cases = []
    for item in raw:
        cases.append({
            "id": item["id"],
            "category": item["category"],
            "query": item.get("query", item.get("question")),
            "expected_chunk_ids": item.get("expected_chunk_ids", item.get("gold_chunk_ids", [])),
            "expected_behavior": item.get("expected_behavior", "retrieve"),
        })
    return cases


def precision_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    if not expected:
        return 0.0
    return sum(cid in set(expected) for cid in retrieved[:k]) / k


def recall_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    if not expected:
        return 0.0
    return sum(cid in retrieved[:k] for cid in expected) / len(expected)


def hit_at_k(retrieved: list[str], expected: list[str], k: int) -> int:
    if not expected:
        return 0
    return int(any(cid in retrieved[:k] for cid in expected))


def evaluate(args: argparse.Namespace) -> dict:
    tests = load_cases(args.test_set)
    retriever = HybridRetriever(chunks_path=args.chunks, chroma_dir=args.chroma)

    rows: list[dict] = []
    explainability: list[dict] = []
    oos_hits: list[dict] = []                 # out-of-scope max-similarity report

    for case in tests:
        query = case["query"]
        expected = case["expected_chunk_ids"]

        # Out-of-scope: NOT scored on precision/recall (no gold). Instead capture
        # the highest cosine similarity any chunk got — a real refusal system
        # should sit BELOW this floor to safely deny fabricated answers.
        if case["category"] == "out_of_scope":
            top = retriever.cosine_search(query, k=3)
            oos_hits.append({
                "id": case["id"],
                "query": query,
                "top_cosine_similarities": [round(s, 4) for _, s in top],
                "top_chunk_ids": [c for c, _ in top],
            })
            continue

        # In-scope: score all three retrievers
        start = time.perf_counter(); semantic = retriever.cosine_search(query, args.pool); semantic_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter(); lexical  = retriever.bm25_search(query, args.pool);   lexical_ms  = (time.perf_counter() - start) * 1000
        start = time.perf_counter(); hybrid   = retriever.hybrid_search(query, args.k, args.w_semantic, args.w_lexical, pool=args.pool); hybrid_ms = (time.perf_counter() - start) * 1000

        sem_ids = [x[0] for x in semantic]
        lex_ids = [x[0] for x in lexical]
        hyb_ids = [h["chunk_id"] for h in hybrid]

        rows.append({
            "id": case["id"], "category": case["category"], "query": query,
            "expected_chunk_ids": json.dumps(expected),
            "expected_behavior": case["expected_behavior"],
            "semantic_p_at_k":  round(precision_at_k(sem_ids, expected, args.k), 4),
            "bm25_p_at_k":      round(precision_at_k(lex_ids, expected, args.k), 4),
            "hybrid_p_at_k":    round(precision_at_k(hyb_ids, expected, args.k), 4),
            "semantic_r_at_k":  round(recall_at_k(sem_ids, expected, args.k), 4),
            "bm25_r_at_k":      round(recall_at_k(lex_ids, expected, args.k), 4),
            "hybrid_r_at_k":    round(recall_at_k(hyb_ids, expected, args.k), 4),
            "semantic_hit_at_k": hit_at_k(sem_ids, expected, args.k),
            "bm25_hit_at_k":    hit_at_k(lex_ids, expected, args.k),
            "hybrid_hit_at_k":  hit_at_k(hyb_ids, expected, args.k),
            "semantic_latency_ms": round(semantic_ms, 2),
            "bm25_latency_ms":     round(lexical_ms, 2),
            "hybrid_latency_ms":   round(hybrid_ms, 2),
        })
        explainability.append({
            "query_id": case["id"], "query": query,
            "expected_chunk_ids": expected, "hits": hybrid,
        })

    # Summary — macro-average over in-scope questions
    def avg(field: str) -> float:
        return round(sum(r[field] for r in rows) / max(1, len(rows)), 4)

    summary = {
        "k": args.k,
        "pool": args.pool,
        "w_semantic": args.w_semantic,
        "w_lexical": args.w_lexical,
        "model": retriever.model_name,
        "num_scored_questions": len(rows),
        "num_out_of_scope": len(oos_hits),
        f"semantic_precision_at_{args.k}": avg("semantic_p_at_k"),
        f"bm25_precision_at_{args.k}":     avg("bm25_p_at_k"),
        f"hybrid_precision_at_{args.k}":   avg("hybrid_p_at_k"),
        f"semantic_recall_at_{args.k}":    avg("semantic_r_at_k"),
        f"bm25_recall_at_{args.k}":        avg("bm25_r_at_k"),
        f"hybrid_recall_at_{args.k}":      avg("hybrid_r_at_k"),
        f"semantic_hit_at_{args.k}":       avg("semantic_hit_at_k"),
        f"bm25_hit_at_{args.k}":           avg("bm25_hit_at_k"),
        f"hybrid_hit_at_{args.k}":         avg("hybrid_hit_at_k"),
        "oos_max_cosine": max((h["top_cosine_similarities"][0] for h in oos_hits), default=None),
    }
    return {
        "summary": summary,
        "rows": rows,
        "explainability": explainability,
        "out_of_scope": oos_hits,
    }


def sweep_k(args: argparse.Namespace) -> list[dict]:
    """Precision@k across k in {1,2,3,5,10} so we can pick the elbow."""
    curve = []
    for k in (1, 2, 3, 5, 10):
        args.k = k
        r = evaluate(args)
        s = r["summary"]
        curve.append({
            "k": k,
            "semantic_p": s[f"semantic_precision_at_{k}"],
            "bm25_p":     s[f"bm25_precision_at_{k}"],
            "hybrid_p":   s[f"hybrid_precision_at_{k}"],
            "hybrid_hit": s[f"hybrid_hit_at_{k}"],
        })
    return curve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-set", default="eval/eval_set.json")
    parser.add_argument("--chunks",   default="corpus/chunks/all_chunks.jsonl")
    parser.add_argument("--chroma",   default="corpus/chunks/chroma_db")
    parser.add_argument("--output-dir", default="artifacts/day2")
    parser.add_argument("--k",         type=int, default=5)
    parser.add_argument("--pool",      type=int, default=50)
    parser.add_argument("--w-semantic", type=float, default=0.7)
    parser.add_argument("--w-lexical",  type=float, default=0.3)
    parser.add_argument("--sweep-k", action="store_true",
                        help="Also compute P@k for k in {1,2,3,5,10} and write a topk_curve.csv")
    args = parser.parse_args()

    result = evaluate(args)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "evaluation.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=result["rows"][0].keys()); w.writeheader(); w.writerows(result["rows"])
    (out / "explainability.json").write_text(json.dumps(result["explainability"], ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "out_of_scope.json").write_text(json.dumps(result["out_of_scope"], ensure_ascii=False, indent=2), encoding="utf-8")

    if args.sweep_k:
        curve = sweep_k(args)
        with (out / "topk_curve.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(curve[0].keys())); w.writeheader(); w.writerows(curve)
        print("Top-K curve:")
        print(json.dumps(curve, indent=2))

    print("Summary:")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
