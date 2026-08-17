"""Day 2 retrieval utilities for Kidney-RAG.

The module keeps the frozen chunk contract and exposes ranked hits containing
exact text, scores, and source metadata for explainability.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

CHUNKS_PATH = Path("corpus/chunks/all_chunks.jsonl")
CHROMA_DIR = "corpus/chunks/chroma_db"
COLLECTION_NAME = "ckd_guidelines"
EMBED_MODEL_NAME = "abhinand/MedEmbed-large-v0.1"
QUERY_INSTRUCTION = "Represent this medical question for retrieving relevant clinical guideline passages: "
MINIMAL_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "is", "are", "and", "or", "for",
    "on", "with", "as", "at", "by", "be", "this", "that", "these", "those",
}
TOKEN_RE = re.compile(r"[a-z0-9]+|[≥≤]")


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in MINIMAL_STOPWORDS]


class HybridRetriever:
    def __init__(self, chunks_path: str | Path = CHUNKS_PATH,
                 chroma_dir: str = CHROMA_DIR,
                 collection_name: str = COLLECTION_NAME,
                 model_name: str = EMBED_MODEL_NAME):
        import chromadb
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer

        self.chunks = [json.loads(line) for line in Path(chunks_path).read_text(encoding="utf-8").splitlines() if line.strip()]
        self.by_id = {c["chunk_id"]: c for c in self.chunks}
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self.client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = self.client.get_collection(collection_name)
        if self.collection.count() != len(self.chunks):
            raise ValueError(f"Chroma has {self.collection.count()} vectors but chunks file has {len(self.chunks)} rows")
        self.embed_model = SentenceTransformer(model_name)
        self.model_name = model_name

    def cosine_search(self, query: str, k: int = 50) -> list[tuple[str, float]]:
        q = self.embed_model.encode([QUERY_INSTRUCTION + query], normalize_embeddings=True, convert_to_numpy=True)
        result = self.collection.query(query_embeddings=q.tolist(), n_results=min(k, len(self.chunks)))
        return [(cid, float(1.0 - distance)) for cid, distance in zip(result["ids"][0], result["distances"][0])]

    def bm25_search(self, query: str, k: int = 50) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        indices = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i]["chunk_id"], float(scores[i])) for i in indices]

    def hybrid_search(self, query: str, k: int = 5, w_semantic: float = 0.7,
                      w_lexical: float = 0.3, k_rrf: int = 60,
                      pool: int = 50) -> list[dict[str, Any]]:
        semantic = self.cosine_search(query, pool)
        lexical = self.bm25_search(query, pool)
        semantic_rank = {cid: rank for rank, (cid, _) in enumerate(semantic, 1)}
        lexical_rank = {cid: rank for rank, (cid, _) in enumerate(lexical, 1)}
        semantic_score = dict(semantic)
        lexical_score = dict(lexical)
        fused: dict[str, float] = {}
        for cid in set(semantic_rank) | set(lexical_rank):
            fused[cid] = (
                w_semantic / (k_rrf + semantic_rank[cid]) if cid in semantic_rank else 0.0
            ) + (
                w_lexical / (k_rrf + lexical_rank[cid]) if cid in lexical_rank else 0.0
            )
        hits = []
        for cid, fused_score in sorted(fused.items(), key=lambda item: item[1], reverse=True)[:k]:
            c = self.by_id[cid]
            hits.append({
                "rank": len(hits) + 1,
                "chunk_id": cid,
                "fused_score": round(float(fused_score), 8),
                "cosine_rank": semantic_rank.get(cid),
                "bm25_rank": lexical_rank.get(cid),
                "cosine_sim": None if cid not in semantic_score else round(semantic_score[cid], 6),
                "bm25_score": None if cid not in lexical_score else round(lexical_score[cid], 6),
                "document_name": c["document_name"],
                "section_title": c["section_title"],
                "page_number": c["page_number"],
                "page_range": c.get("page_range", [c["page_number"], c["page_number"]]),
                "source_url": c["source_url"],
                "text": c["text"],
            })
        return hits
