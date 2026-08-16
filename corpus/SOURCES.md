# CKD RAG Corpus — Source Manifest

**Topic:** Adult Chronic Kidney Disease (CKD) — Assessment, Management & Screening
**Rule:** Official, public guideline PDFs only (NICE, KDIGO, USPSTF, CDC). No private / credential-gated data.
**Collected:** 2026-08-16 (Day 1). Location: `corpus/raw_pdfs/`.

| # | Document | Publisher | Pages | Size | Topic | Text |
|---|----------|-----------|-------|------|-------|------|
| 1 | Chronic kidney disease: assessment and management — **NICE NG203** | NICE (UK) | 78 | 0.41 MB | Management | ✅ extractable |
| 2 | **KDIGO 2024** CKD Guideline — full | KDIGO | 199 | 5.97 MB | Management | ✅ extractable |
| 3 | KDIGO 2024 CKD Guideline — Summary of Recommendations & Practice Points | KDIGO | 21 | 0.36 MB | Management | ✅ extractable |
| 4 | KDIGO 2024 CKD Guideline — Executive Summary | KDIGO | 18 | 0.89 MB | Management | ✅ extractable |
| 5 | Screening for CKD — **USPSTF Recommendation Statement** (I-statement) | USPSTF (US) | 6 | 0.14 MB | Screening | ✅ extractable |

**Total collected: 5 PDFs · 322 pages · all machine-readable (no OCR needed).**

## Source URLs
1. NICE NG203 — https://www.nice.org.uk/guidance/ng203/resources/chronic-kidney-disease-assessment-and-management-pdf-66143713055173
2. KDIGO 2024 full — https://kdigo.org/wp-content/uploads/2024/03/KDIGO-2024-CKD-Guideline.pdf
3. KDIGO 2024 summary — https://kdigo.org/wp-content/uploads/2026/05/KDIGO-2024-CKD-Guideline-Summary-Recommendations-and-Practice-Points.pdf
4. KDIGO 2024 exec summary — https://kdigo.org/wp-content/uploads/2017/02/KDIGO-2024-CKD-Guideline-Executive-Summary.pdf
5. USPSTF CKD screening — https://www.uspreventiveservicestaskforce.org/home/getfilebytoken/ZRz9nTrjKkRtNTe6hgPze-

## Pending / optional (bot-blocked — needs manual browser download)
- **CDC — Chronic Kidney Disease in the United States (Fact Sheet)** — https://www.cdc.gov/kidney-disease/media/pdfs/CKD-Factsheet-H.pdf
  CDC serves behind Akamai bot protection (HTTP 403 to scripted requests). Patient-facing epidemiology, not a clinical guideline — lowest priority. Download manually in a browser and drop into `raw_pdfs/` if wanted.

## Notes for the pipeline
- **USPSTF is an I-statement** ("evidence is insufficient" to assess screening in asymptomatic adults). This is an ideal test case for the **"Insufficient Evidence" confidence label** and the **safe-refusal demo (Case C)**.
- NICE NG203 and KDIGO 2024 overlap on management → good for **multi-chunk synthesis** test cases (Case B) and cross-source citation.
- Every chunk must carry: `document_name, page_number, section_title, chunk_id, source_url` (agenda metadata schema).
