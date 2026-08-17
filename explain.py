"""Minimal Day 2 explainability view: retrieval evidence before generation."""
from __future__ import annotations

import argparse
import json
from retrieval import HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--pool", type=int, default=50)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    retriever = HybridRetriever()
    hits = retriever.hybrid_search(args.query, k=args.k, pool=args.pool)
    if args.json:
        print(json.dumps({"query": args.query, "hits": hits}, ensure_ascii=False, indent=2))
        return
    print(f"Query: {args.query}\n")
    for hit in hits:
        print(f"[{hit['rank']}] {hit['chunk_id']} | fused={hit['fused_score']:.6f} | cosine={hit['cosine_sim']} | bm25={hit['bm25_score']}")
        print(f"    {hit['document_name']} | {hit['section_title']} | page {hit['page_number']} | source: {hit['source_url']}")
        print(f"    {hit['text']}\n")


if __name__ == "__main__":
    main()
