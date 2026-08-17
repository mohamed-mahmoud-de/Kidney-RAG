"""Chunk-size / overlap ablation (deck requirement).

For each of 3 configurations, rebuild the chunk set from the parsed pages,
embed with a SMALL FAST model, and score Precision@k / Recall@k / Hit@k on
eval/eval_set.json. The purpose is RELATIVE comparison of chunk configs —
not absolute model quality — so a fast embedder (all-MiniLM-L6-v2, 384-dim)
keeps the whole ablation under ~10 minutes on CPU.

Important: this DOES NOT touch the main index (corpus/chunks/chroma_db) or
all_chunks.jsonl. Everything is in-memory + models/ cache.

The eval set's gold chunk_ids are pegged to the current production chunker
(550/55). When we re-chunk at a different size, chunk_ids won't match by
name, so we score on a fallback: "does the top-k for the same query CONTAIN
one of the source pages of the original gold chunks?" This is the standard
way to compare chunk configs without a per-config re-labeled eval set.

Usage:
    python ablation.py
"""
from __future__ import annotations

import argparse
import json
import re
import time
import warnings
from pathlib import Path

import numpy as np
import tiktoken

warnings.filterwarnings("ignore")


# The 3 configs to compare — inside the deck's 400-800 recommended range.
CONFIGS = [
    {"slug": "400_40",  "chunk_size": 400, "overlap": 40,  "hard_max": 550},
    {"slug": "550_55",  "chunk_size": 550, "overlap": 55,  "hard_max": 750},   # current production
    {"slug": "700_70",  "chunk_size": 700, "overlap": 70,  "hard_max": 900},
]

# Fast local model for relative comparison
ABLATION_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = Path("models")

PARSED_DIR = Path("corpus/parsed")
PARSED_FILES = [
    ("kdigo",    "kdigo_parsed.json",        set(range(34, 54))),
    ("nice",     "nice_parsed.json",          set()),
    ("kdigo_dm", "kdigo_diabetes_parsed.json", set(range(20, 30))),
    ("uspstf",   "uspstf_parsed.json",         set()),
]
NON_CLINICAL = {"notice", "acknowledgments", "work group membership"}
def is_clinical(section_title: str) -> bool:
    t = section_title.strip().lower()
    if t in NON_CLINICAL: return False
    if "research recommendation" in t: return False
    if "members of the u.s. preventive" in t: return False
    return True

ENC = tiktoken.get_encoding("cl100k_base")
def n_tokens(s): return len(ENC.encode(s))


# ---------- chunker (self-contained, mirrors chunker.ipynb logic) ----------
SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9•\-])')

def overlap_tail(text, n_overlap):
    toks = ENC.encode(text)
    if len(toks) <= n_overlap: return text
    tail = ENC.decode(toks[-n_overlap:])
    m = re.search(r'[.!?]\s+([A-Z])', tail)
    if m: tail = tail[m.start()+2:]
    return tail.strip()

def split_long(block, chunk_size):
    sents = SENTENCE_SPLIT.split(block)
    out, cur, cur_t = [], [], 0
    for s in sents:
        st = n_tokens(s)
        if st > chunk_size:
            if cur: out.append(" ".join(cur)); cur, cur_t = [], 0
            toks = ENC.encode(s)
            for i in range(0, len(toks), chunk_size):
                out.append(ENC.decode(toks[i:i+chunk_size]))
            continue
        if cur_t + st <= chunk_size: cur.append(s); cur_t += st
        else: out.append(" ".join(cur)); cur, cur_t = [s], st
    if cur: out.append(" ".join(cur))
    return out

def split_section(text, chunk_size, overlap, hard_max):
    blocks = [b.strip() for b in re.split(r'\n\s*\n', text) if b.strip()]
    if not blocks: return []
    normalized = []
    for b in blocks:
        if n_tokens(b) > hard_max: normalized.extend(split_long(b, chunk_size))
        else: normalized.append(b)
    chunks, cur, cur_t = [], [], 0
    for block in normalized:
        bt = n_tokens(block)
        if cur_t + bt <= chunk_size:
            cur.append(block); cur_t += bt; continue
        if cur:
            ct = "\n\n".join(cur); chunks.append(ct)
            tail = overlap_tail(ct, overlap)
            cur = [tail, block] if tail else [block]
            cur_t = n_tokens("\n\n".join(cur))
        else:
            chunks.append(block); cur, cur_t = [], 0
    if cur: chunks.append("\n\n".join(cur))
    if len(chunks) >= 2 and n_tokens(chunks[-1]) < 100:
        last = chunks.pop()
        chunks[-1] = chunks[-1] + "\n\n" + last
    return chunks


def build_chunks_for_config(cfg):
    """Full chunk build for one config. Returns list of chunk dicts (in memory)."""
    all_chunks = []
    for short, fname, skip_pages in PARSED_FILES:
        pages = json.loads((PARSED_DIR / fname).read_text(encoding="utf-8"))
        pages = [p for p in pages if p["page_number"] not in skip_pages]

        # group consecutive pages by section
        groups = []
        for p in pages:
            if groups and groups[-1]["section_title"] == p["section_title"]:
                g = groups[-1]
                g["page_end"] = p["page_number"]
                g["text"] += "\n\n" + p["text"]
            else:
                groups.append({
                    "document_name": p["document_name"],
                    "source_url": p["source_url"],
                    "section_title": p["section_title"],
                    "page_start": p["page_number"],
                    "page_end":   p["page_number"],
                    "text": p["text"],
                })

        per_page_counter = {}
        for g in groups:
            if not is_clinical(g["section_title"]): continue
            for piece in split_section(g["text"], cfg["chunk_size"], cfg["overlap"], cfg["hard_max"]):
                ps = g["page_start"]
                per_page_counter[ps] = per_page_counter.get(ps, 0) + 1
                idx = per_page_counter[ps]
                all_chunks.append({
                    "chunk_id": f"{short}_p{ps}_c{idx:02d}",
                    "document_name": g["document_name"],
                    "source_url": g["source_url"],
                    "page_number": ps,
                    "page_start": g["page_start"], "page_end": g["page_end"],
                    "section_title": g["section_title"],
                    "text": piece,
                    "token_count": n_tokens(piece),
                })
    return all_chunks


def load_cases():
    payload = json.loads(Path("eval/eval_set.json").read_text(encoding="utf-8"))
    return payload["questions"]


def load_prod_gold_pages():
    """For each in-scope question, look up the (doc-short, page) of each gold chunk
    from the PRODUCTION all_chunks.jsonl. We'll match ablation-config chunks by
    (doc-short, page) since chunk_ids won't align across configs."""
    prod = {}
    for line in Path("corpus/chunks/all_chunks.jsonl").read_text(encoding="utf-8").splitlines():
        c = json.loads(line)
        prod[c["chunk_id"]] = (c["chunk_id"].split("_p")[0], c["page_number"])
    return prod


def evaluate_config(cfg, k):
    print(f"\n=== Config {cfg['slug']}  chunk={cfg['chunk_size']} overlap={cfg['overlap']} ===")
    t0 = time.time()
    chunks = build_chunks_for_config(cfg)
    print(f"  built {len(chunks)} chunks in {time.time()-t0:.1f}s")

    # Embed with the fast ablation model (cache per config)
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / f"ablation_{cfg['slug']}.npy"
    if cache.exists():
        vecs = np.load(cache)
        if vecs.shape[0] == len(chunks):
            print(f"  [cache] reused {cache}")
        else:
            print(f"  [cache stale] re-embedding")
            vecs = None
    else:
        vecs = None

    if vecs is None:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(ABLATION_MODEL)
        print(f"  [embed] {len(chunks)} chunks with {ABLATION_MODEL} ...")
        t0 = time.time()
        vecs = m.encode([c["text"] for c in chunks], batch_size=64, show_progress_bar=True,
                        normalize_embeddings=True, convert_to_numpy=True)
        print(f"  [embed] done in {time.time()-t0:.1f}s")
        np.save(cache, vecs)

    # Build (doc_short, page) lookup for ablation chunks
    ab_ref = [(c["chunk_id"].split("_p")[0], c["page_number"]) for c in chunks]

    # Score
    cases = load_cases()
    in_scope = [c for c in cases if c["category"] != "out_of_scope"]

    prod_map = load_prod_gold_pages()

    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(ABLATION_MODEL)
    query_vecs = m.encode([c["query"] if "query" in c else c["question"] for c in in_scope],
                          normalize_embeddings=True, convert_to_numpy=True)

    p_scores, r_scores, h_scores = [], [], []
    for c, qv in zip(in_scope, query_vecs):
        # gold: set of (doc, page) that this question's gold chunks cover
        gold_pages = set()
        for gid in c["gold_chunk_ids"]:
            if gid in prod_map:
                gold_pages.add(prod_map[gid])

        sims = vecs @ qv
        idx = np.argsort(sims)[::-1][:k]
        got_pages = [ab_ref[i] for i in idx]

        # A "hit" here is: the retrieved chunk is on one of the gold pages
        # (correct doc + correct page). Standard proxy for cross-config eval.
        hits_in_topk = sum(1 for gp in got_pages if gp in gold_pages)
        p = hits_in_topk / k
        r = (sum(1 for g in gold_pages if g in got_pages) / len(gold_pages)) if gold_pages else 0.0
        h = int(any(gp in gold_pages for gp in got_pages))
        p_scores.append(p); r_scores.append(r); h_scores.append(h)

    return {
        "config": cfg["slug"],
        "chunk_size": cfg["chunk_size"],
        "overlap": cfg["overlap"],
        "hard_max": cfg["hard_max"],
        "n_chunks": len(chunks),
        "median_tokens": int(np.median([c["token_count"] for c in chunks])),
        "k": k,
        "n_scored": len(in_scope),
        "precision_at_k": round(np.mean(p_scores), 4),
        "recall_at_k":    round(np.mean(r_scores), 4),
        "hit_at_k":       round(np.mean(h_scores), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--output-dir", default="artifacts/day2")
    args = ap.parse_args()

    print("Chunk-size ablation")
    print(f"  Ablation model: {ABLATION_MODEL}  (fast; for relative comparison only)")
    print(f"  k={args.k}   Configs: {[c['slug'] for c in CONFIGS]}")
    print("  Scoring: page-level (gold pages retrieved in top-k) because chunk_ids")
    print("           don't align across chunk sizes.")

    results = [evaluate_config(c, args.k) for c in CONFIGS]

    print("\n" + "=" * 82)
    print(f"CHUNK-SIZE ABLATION @ k={args.k}  (page-level scoring on {results[0]['n_scored']} in-scope Qs)")
    print("=" * 82)
    print(f"{'Config':<10} {'Chunks':>8} {'MedTok':>8} {'P@k':>8} {'R@k':>8} {'Hit@k':>8}")
    for r in results:
        print(f"{r['config']:<10} {r['n_chunks']:>8} {r['median_tokens']:>8} {r['precision_at_k']:>8.4f} {r['recall_at_k']:>8.4f} {r['hit_at_k']:>8.4f}")

    winner = max(results, key=lambda r: (r["hit_at_k"], r["precision_at_k"]))
    print(f"\nWinner (by Hit@k then P@k): {winner['config']}")

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "chunk_ablation.json").write_text(
        json.dumps({"k": args.k, "ablation_model": ABLATION_MODEL,
                    "results": results, "winner": winner["config"]},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWritten: {out / 'chunk_ablation.json'}")


if __name__ == "__main__":
    main()
