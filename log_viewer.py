"""View and summarize the Kidney-RAG query log.

Usage:
    python log_viewer.py              # print summary stats
    python log_viewer.py --full       # print every log entry
    python log_viewer.py --export     # export logs/query_log_report.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

LOG_PATH = Path(__file__).parent / "logs" / "query_log.jsonl"
REPORT_PATH = Path(__file__).parent / "logs" / "query_log_report.json"


def load_entries() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    entries = []
    with open(LOG_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def print_summary(entries: list[dict]) -> None:
    if not entries:
        print("No queries logged yet.")
        return

    total = len(entries)
    passed = sum(1 for e in entries if e["gate_passed"])
    refused = total - passed
    high_conf = sum(1 for e in entries if e["confidence"] == "high")
    medium_conf = sum(1 for e in entries if e["confidence"] == "medium")
    format_ok = sum(1 for e in entries if e["format_valid"] is True)
    format_fail = sum(1 for e in entries if e["format_valid"] is False)

    docs_seen = {}
    for e in entries:
        for c in e.get("chunks_used", []):
            doc = c["document_name"]
            docs_seen[doc] = docs_seen.get(doc, 0) + 1

    print(f"{'='*60}")
    print(f"  Kidney-RAG Query Log Summary")
    print(f"{'='*60}")
    print(f"  Total queries:       {total}")
    print(f"  Gate PASSED:         {passed}")
    print(f"  Gate REFUSED:        {refused}")
    print(f"  Confidence HIGH:     {high_conf}")
    print(f"  Confidence MEDIUM:   {medium_conf}")
    print(f"  Format valid:        {format_ok}")
    print(f"  Format invalid:      {format_fail}")
    print(f"{'─'*60}")
    print(f"  Documents referenced:")
    for doc, count in sorted(docs_seen.items(), key=lambda x: -x[1]):
        print(f"    {doc}: {count} chunk hits")
    print(f"{'─'*60}")
    print(f"  Time range: {entries[0]['timestamp']} → {entries[-1]['timestamp']}")
    print(f"{'='*60}")


def print_full(entries: list[dict]) -> None:
    if not entries:
        print("No queries logged yet.")
        return

    for i, e in enumerate(entries, 1):
        status = "PASS" if e["gate_passed"] else "REFUSED"
        fmt = e.get("format_valid")
        fmt_str = "OK" if fmt is True else ("FAIL" if fmt is False else "n/a")
        print(f"\n{'='*70}")
        print(f"  [{i}] {e['timestamp']}")
        print(f"  Query:      {e['query']}")
        print(f"  Gate:       {status}  |  Confidence: {e['confidence']}  |  Format: {fmt_str}")
        print(f"  Backend:    {e['backend']} ({e['model']})")
        if e["top_chunk_id"]:
            print(f"  Top hit:    {e['top_chunk_id']} (cosine={e['top_cosine_sim']})")
            print(f"              {e['top_document']} — {e['top_section']}")
        if e["refused"]:
            print(f"  Reason:     {e['gate_reason']}")
        chunks = e.get("chunks_used", [])
        if chunks:
            print(f"  Chunks ({len(chunks)}):")
            for c in chunks:
                print(f"    {c['chunk_id']:25s}  cos={c['cosine_sim']:.4f}  {c['document_name']}")


def export_report(entries: list[dict]) -> None:
    if not entries:
        print("No queries to export.")
        return

    report = {
        "generated_at": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "total_queries": len(entries),
        "gate_passed": sum(1 for e in entries if e["gate_passed"]),
        "gate_refused": sum(1 for e in entries if not e["gate_passed"]),
        "confidence_breakdown": {
            "high": sum(1 for e in entries if e["confidence"] == "high"),
            "medium": sum(1 for e in entries if e["confidence"] == "medium"),
            "insufficient": sum(1 for e in entries if e["confidence"] == "insufficient"),
        },
        "format_breakdown": {
            "valid": sum(1 for e in entries if e["format_valid"] is True),
            "invalid": sum(1 for e in entries if e["format_valid"] is False),
            "not_checked": sum(1 for e in entries if e["format_valid"] is None),
        },
        "entries": entries,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Exported {len(entries)} entries to {REPORT_PATH}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    entries = load_entries()

    if "--export" in sys.argv:
        export_report(entries)
    elif "--full" in sys.argv:
        print_full(entries)
    else:
        print_summary(entries)
