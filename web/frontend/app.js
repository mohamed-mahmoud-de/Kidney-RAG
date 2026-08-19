/* Kidney-RAG frontend controller.
 *
 * Small, dependency-free client for the FastAPI backend. Handles:
 *   - question submission + loading state
 *   - answer rendering (formatted, with clickable chunk_id → source citation)
 *   - retrieved-chunks accordion
 *   - confidence + evidence-strength + faithfulness badges
 *   - graceful error paths (network, quota, backend not ready)
 */

const $ = (id) => document.getElementById(id);

const els = {
  form: $("ask-form"),
  input: $("q"),
  btn: $("ask-btn"),
  status: $("status"),
  result: $("result"),
  resultQ: $("result-q"),
  badgeStrength: $("badge-strength"),
  badgeConf: $("badge-conf"),
  badgeCosine: $("badge-cosine"),
  badgeServed: $("badge-served"),
  mFaith: $("m-faith"),
  mCite: $("m-cite"),
  mClaims: $("m-claims"),
  answer: $("answer"),
  unsupported: $("unsupported"),
  hits: $("hits"),
  sourcesCount: $("sources-count"),
  sourceGrid: $("source-grid"),
  footerModel: $("footer-model"),
};

// -------- API --------

async function apiAsk(question) {
  const res = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, k: 5 }),
  });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (body && body.detail) detail = body.detail;
    } catch { /* not json */ }
    throw new Error(detail);
  }
  return res.json();
}

async function apiHealth() {
  try { return await (await fetch("/health")).json(); }
  catch { return null; }
}
async function apiSources() {
  try { return await (await fetch("/api/sources")).json(); }
  catch { return null; }
}

// -------- Utilities --------

const escapeHtml = (s) => (s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function fmtPct(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return `${Math.round(v * 100)}%`;
}

function fmtCosine(v) {
  if (v === null || v === undefined) return "—";
  return v.toFixed(3);
}

function setBadge(el, level, textOverride) {
  el.className = "badge";
  const known = ["strong", "partial", "weak", "insufficient"];
  if (level && known.includes(level)) el.classList.add(level);
  el.textContent = textOverride ?? level ?? "—";
}

// Render the LLM answer text with:
//   - "### 1. Recommendation" style headings turned into <h3>
//   - excerpt quotes ("...") turned into <blockquote>
//   - the [Source: ... | chunk_id:X | url] tags into clickable cite chips
//   - the trailing "---\n*disclaimer*" turned into a subtle inline footer
function renderAnswer(text) {
  if (!text) return "";
  const CITATION_RE = /\[Source:\s*([^—]+?)\s*—\s*(.+?),\s*(pp?\.\d+(?:[–-]\d+)?)\s*\|\s*chunk_id:(\S+?)\s*\|\s*([^\]]+?)\]/g;

  // Split on trailing disclaimer marker if present.
  let body = text;
  let disclaimer = "";
  const discIdx = text.lastIndexOf("\n---");
  if (discIdx > -1) {
    body = text.slice(0, discIdx);
    disclaimer = text.slice(discIdx + 4).trim().replace(/^\*|\*$/g, "").trim();
  }

  // Strip markdown code fences the LLM sometimes wraps excerpts in
  // (```plaintext ... ```). Keep the content, drop the fences —
  // clinical excerpts should read as prose, not code.
  body = body.replace(/```[a-zA-Z]*\n?/g, "").replace(/```/g, "");

  // Also strip stray single backticks that don't wrap real code.
  body = body.replace(/`([^`\n]+)`/g, "$1");

  // Replace citations FIRST so we can style them independently.
  const withCites = escapeHtml(body).replace(
    CITATION_RE,
    (_m, doc, section, pages, chunkId, url) => {
      const label = escapeHtml(`${chunkId} · ${pages}`);
      const title = escapeHtml(`${doc} — ${section}`);
      return `<a class="cite" href="${escapeHtml(url)}" target="_blank" rel="noopener" title="${title}">${label}</a>`;
    },
  );

  // Convert markdown-lite headings and quote blocks.
  const lines = withCites.split("\n");
  const out = [];
  let inQuote = false;
  for (const raw of lines) {
    const line = raw.trimEnd();
    const h = /^\s*#{0,3}\s*\d+\.\s*(Recommendation|Excerpt|Citation)s?\s*:?\s*$/i.exec(line);
    if (h) {
      if (inQuote) { out.push("</blockquote>"); inQuote = false; }
      out.push(`<h3>${h[1]}</h3>`);
      continue;
    }
    // Bare quoted paragraph → blockquote (very light heuristic).
    if (/^\s*&quot;.*&quot;\s*$/.test(line) || /^\s*"[^"]+"\s*$/.test(line.replace(/&quot;/g, '"'))) {
      if (!inQuote) { out.push("<blockquote>"); inQuote = true; }
      out.push(line.trim());
      continue;
    }
    if (inQuote) { out.push("</blockquote>"); inQuote = false; }
    out.push(line);
  }
  if (inQuote) out.push("</blockquote>");

  let html = out.join("\n");
  if (disclaimer) {
    html += `<span class="disclaimer-inline">${escapeHtml(disclaimer)}</span>`;
  }
  return html;
}

// Collapse single-word-per-line runs (PDF wrapping artefact from KDIGO
// chapter/table pages) into flowing prose. Preserves paragraph breaks
// (any run of 2+ newlines becomes exactly one blank line).
function normalizeChunkText(raw) {
  if (!raw) return "";
  // Split on blank lines → paragraphs.
  const paragraphs = raw.split(/\n{2,}/).map((para) => {
    // Within a paragraph, collapse single newlines to spaces (PDF wrapping).
    return para.replace(/\s*\n\s*/g, " ").replace(/[ \t]{2,}/g, " ").trim();
  }).filter(Boolean);
  return paragraphs.join("\n\n");
}

function renderHit(hit, idx) {
  const pages = hit.page_range && hit.page_range[0] !== hit.page_range[1]
    ? `pp.${hit.page_range[0]}–${hit.page_range[1]}`
    : `p.${hit.page_number}`;
  const cos = hit.cosine_sim === null || hit.cosine_sim === undefined
    ? "—" : hit.cosine_sim.toFixed(4);
  const fused = hit.fused_score === null || hit.fused_score === undefined
    ? "—" : hit.fused_score.toFixed(5);
  return `
    <li class="hit">
      <div class="hit-meta">
        <div>
          <div class="hit-title">${escapeHtml(`#${idx + 1} · ${hit.document_name}`)}</div>
          <div class="hit-sub">${escapeHtml(hit.section_title || "n/a")} · ${escapeHtml(pages)} · <code>${escapeHtml(hit.chunk_id)}</code></div>
        </div>
        <div class="hit-scores">
          <span>cos <b>${cos}</b></span>
          <span>rrf <b>${fused}</b></span>
        </div>
      </div>
      <p class="hit-text">${escapeHtml(normalizeChunkText(hit.text))}</p>
      <div class="hit-actions">
        <a href="${escapeHtml(hit.source_url)}" target="_blank" rel="noopener">Open source guideline ↗</a>
      </div>
    </li>`;
}

// -------- Controller --------

function setLoading(loading) {
  els.btn.disabled = loading;
  els.btn.setAttribute("aria-busy", loading ? "true" : "false");
}

function setStatus(msg, isError = false) {
  els.status.textContent = msg || "";
  els.status.classList.toggle("error", !!isError);
}

async function handleAsk(question) {
  setStatus("Retrieving from indexed guidelines…");
  setLoading(true);
  els.result.hidden = true;

  try {
    const data = await apiAsk(question);
    setStatus("");
    renderResult(data);
    els.result.hidden = false;
    els.result.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    setStatus(`Error: ${err.message}`, true);
  } finally {
    setLoading(false);
  }
}

function renderResult(data) {
  els.resultQ.textContent = data.question;

  // Badges
  setBadge(els.badgeStrength, data.evidence_strength);
  const conf = data.refused ? "insufficient" : data.confidence;
  setBadge(els.badgeConf, conf, `confidence: ${conf}`);
  els.badgeCosine.textContent = `cosine ${fmtCosine(data.top_cosine)}`;
  els.badgeCosine.className = "badge badge-outline";

  // "via <backend>" shows which provider key served this answer — makes
  // failover visible on stage when a Gemini key trips and HF takes over.
  if (data.served_by) {
    els.badgeServed.hidden = false;
    els.badgeServed.textContent = `via ${data.served_by}`;
  } else {
    els.badgeServed.hidden = true;
  }

  // Mini metrics
  const s = data.safety;
  els.mFaith.textContent = s ? fmtPct(s.faithfulness) : "—";
  els.mCite.textContent = s ? fmtPct(s.citation_accuracy) : "—";
  els.mClaims.textContent = s ? `${s.supported_claims}/${s.total_claims}` : "—";

  // Answer body
  els.answer.innerHTML = renderAnswer(data.answer);

  // Unsupported-claim warning banner
  const unsupported = s && s.unsupported_claims && s.unsupported_claims.length > 0
    ? s.unsupported_claims : null;
  if (unsupported) {
    els.unsupported.hidden = false;
    els.unsupported.innerHTML =
      `<strong>${unsupported.length} unsupported claim${unsupported.length > 1 ? "s" : ""} flagged:</strong> ` +
      unsupported.map(escapeHtml).join(" · ");
  } else {
    els.unsupported.hidden = true;
    els.unsupported.textContent = "";
  }

  // Hits
  els.sourcesCount.textContent = String(data.hits.length);
  els.hits.innerHTML = data.hits.map(renderHit).join("");
}

// -------- Wire up --------

els.form.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = els.input.value.trim();
  if (q.length < 3) return;
  handleAsk(q);
});

document.querySelectorAll(".example").forEach((btn) => {
  btn.addEventListener("click", () => {
    const map = {
      "SGLT2 in CKD": "At what eGFR can an SGLT2 inhibitor be started in a patient with type 2 diabetes and CKD?",
      "ACR threshold for A3": "What ACR value defines severely increased albuminuria (category A3)?",
      "First-line BP drug in CKD + proteinuria": "Which drug class is first-line for a patient with CKD, hypertension, and proteinuria?",
      "Should adults be screened for CKD?": "Should asymptomatic adults be screened for chronic kidney disease?",
      "Treatment for acute appendicitis": "How should acute appendicitis be managed?",
    };
    els.input.value = map[btn.textContent] || btn.textContent;
    els.input.focus();
    handleAsk(els.input.value);
  });
});

(async () => {
  const [health, sources] = await Promise.all([apiHealth(), apiSources()]);
  if (health) {
    const parts = [];
    if (health.embed_model) parts.push(health.embed_model.split("/").pop());
    if (health.model) parts.push(health.model);
    if (parts.length) els.footerModel.textContent = parts.join(" + ");
    if (!health.generator_ready) {
      setStatus(
        "Note: the LLM is not configured on this server — you'll see retrieval + refusals, " +
        "but no generated answer text. Ask the operator to set GOOGLE_API_KEY.",
      );
    }
  }
  if (sources && sources.guidelines) {
    els.sourceGrid.innerHTML = sources.guidelines.map((s) => `
      <div class="source">
        <h3>${escapeHtml(s.name)}</h3>
        <div class="source-role">${escapeHtml(s.role)}</div>
        <div class="source-meta">
          <span>${s.pages} pages</span>
          <a href="${escapeHtml(s.url)}" target="_blank" rel="noopener">Open ↗</a>
        </div>
      </div>`).join("");
  }
})();
