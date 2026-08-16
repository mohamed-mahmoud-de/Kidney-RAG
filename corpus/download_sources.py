"""Download official public CKD guideline PDFs for the RAG corpus.
Records size + sha256 + http status into corpus/manifest.json.
Official/public sources only (NICE, KDIGO, USPSTF, CDC)."""
import requests, hashlib, json, os, sys

OUT = os.path.join(os.path.dirname(__file__), "raw_pdfs")
os.makedirs(OUT, exist_ok=True)

# (filename, url, publisher, title, topic_tag)
SOURCES = [
    ("NICE_NG203_CKD_assessment_and_management.pdf",
     "https://www.nice.org.uk/guidance/ng203/resources/chronic-kidney-disease-assessment-and-management-pdf-66143713055173",
     "NICE", "Chronic kidney disease: assessment and management (NG203)", "ckd-management"),
    ("KDIGO_2024_CKD_Guideline_full.pdf",
     "https://kdigo.org/wp-content/uploads/2024/03/KDIGO-2024-CKD-Guideline.pdf",
     "KDIGO", "KDIGO 2024 Clinical Practice Guideline for the Evaluation and Management of CKD (full)", "ckd-management"),
    ("KDIGO_2024_CKD_Guideline_Summary_Recommendations.pdf",
     "https://kdigo.org/wp-content/uploads/2026/05/KDIGO-2024-CKD-Guideline-Summary-Recommendations-and-Practice-Points.pdf",
     "KDIGO", "KDIGO 2024 CKD Guideline - Summary of Recommendations and Practice Points", "ckd-management"),
    ("KDIGO_2024_CKD_Guideline_Executive_Summary.pdf",
     "https://kdigo.org/wp-content/uploads/2017/02/KDIGO-2024-CKD-Guideline-Executive-Summary.pdf",
     "KDIGO", "KDIGO 2024 CKD Guideline - Executive Summary", "ckd-management"),
    ("USPSTF_CKD_Screening_Recommendation.pdf",
     "https://www.uspreventiveservicestaskforce.org/home/getfilebytoken/ZRz9nTrjKkRtNTe6hgPze-",
     "USPSTF", "Screening for Chronic Kidney Disease: USPSTF Recommendation Statement", "ckd-screening"),
    ("CDC_CKD_Factsheet_2026.pdf",
     "https://www.cdc.gov/kidney-disease/media/pdfs/CKD-Factsheet-H.pdf",
     "CDC", "Chronic Kidney Disease in the United States (CDC Fact Sheet)", "ckd-public-health"),
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/pdf,*/*",
}

manifest = []
for fname, url, pub, title, tag in SOURCES:
    dest = os.path.join(OUT, fname)
    rec = {"document_name": fname, "publisher": pub, "title": title,
           "topic_tag": tag, "source_url": url}
    try:
        r = requests.get(url, headers=HEADERS, timeout=90, allow_redirects=True)
        rec["http_status"] = r.status_code
        rec["final_url"] = r.url
        ctype = r.headers.get("Content-Type", "")
        rec["content_type"] = ctype
        is_pdf = r.content[:5] == b"%PDF-"
        rec["is_pdf"] = is_pdf
        if r.status_code == 200 and is_pdf:
            with open(dest, "wb") as f:
                f.write(r.content)
            rec["bytes"] = len(r.content)
            rec["size_mb"] = round(len(r.content) / 1e6, 2)
            rec["sha256"] = hashlib.sha256(r.content).hexdigest()
            rec["status"] = "OK"
        else:
            rec["status"] = "FAILED"
            rec["note"] = f"status={r.status_code} is_pdf={is_pdf} ctype={ctype[:60]}"
    except Exception as e:
        rec["status"] = "ERROR"
        rec["note"] = str(e)[:200]
    print(f"[{rec['status']:6}] {fname}  {rec.get('size_mb','-')}MB  {rec.get('http_status','')}")
    manifest.append(rec)

with open(os.path.join(os.path.dirname(__file__), "manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
print("\nManifest written. OK:", sum(1 for m in manifest if m["status"] == "OK"), "/", len(manifest))
