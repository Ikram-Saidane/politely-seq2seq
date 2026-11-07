/* Politely — front end. Plain JavaScript, no framework.
 * Talks to the FastAPI back end: /api/info, /api/rewrite, /api/compare, /api/dashboard.
 * Every chart is drawn by hand as SVG from the JSON the server returns (no chart library). */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x, d = 0) => (x == null ? "—" : (100 * x).toFixed(d) + "%");
const num = (x, d = 2) => (x == null ? "—" : Number(x).toFixed(d));

const state = { info: null, dash: null, decoding: "beam", split: "test", maxChars: 300 };
const SERIES = ["#3d6fb0", "#b4532a", "#176f63", "#d9a441", "#7a4fa3", "#3f8f4f", "#c2476b", "#6b7a77"];

/* ------------------------------------------------------------------ API helper */
async function api(path, body) {
  const opts = body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {};
  let res;
  try { res = await fetch(path, opts); }
  catch { throw new Error("Cannot reach the server. Is it still running? (python run_app.py)"); }
  let data = null;
  try { data = await res.json(); } catch { /* not JSON */ }
  if (!res.ok) {
    let msg = data && data.detail;
    if (Array.isArray(msg)) msg = msg.map((d) => d.msg).join(" ");   // pydantic validation errors
    throw new Error(msg || `The server answered with an error (${res.status}).`);
  }
  return data;
}

function toast(msg, bad = false) {
  const t = $("#toast");
  t.textContent = msg; t.classList.toggle("bad", bad); t.hidden = false;
  clearTimeout(toast._t); toast._t = setTimeout(() => (t.hidden = true), 3200);
}

/* ------------------------------------------------------------------ theme */
function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem("politely-theme"); } catch { /* storage unavailable */ }
  if (saved) document.documentElement.dataset.theme = saved;
  $("#themeBtn").addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("politely-theme", next); } catch { /* ignore */ }
    if (state.dash) renderDashboard();     // charts read colours from CSS
  });
}

/* ------------------------------------------------------------------ tabs */
function showTab(name) {
  $$(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
  $$(".panel").forEach((p) => (p.hidden = p.id !== `panel-${name}`));
  if (location.hash !== `#${name}`) history.replaceState(null, "", `#${name}`);
  if (name === "performance" || name === "dataset") { if (!state.dash) loadDashboard(); else renderDashboard(); }
  window.scrollTo({ top: 0 });
}
function initTabs() {
  $$(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
  const h = location.hash.slice(1);
  if ($(`#panel-${h}`)) showTab(h);
}

/* ------------------------------------------------------------------ rewrite tab */
function setInputError(msg, el = $("#inputText"), box = $("#inputError")) {
  box.textContent = msg || ""; box.hidden = !msg;
  el.setAttribute("aria-invalid", msg ? "true" : "false");
}

function clientCheck(text) {
  const t = text.trim();
  if (!t) return "Please type a sentence to rewrite.";
  if (t.length > state.maxChars) return `Please keep it under ${state.maxChars} characters.`;
  if (!/\p{L}/u.test(t)) return "The sentence must contain words, not only numbers or symbols.";
  return null;
}

function initRewrite() {
  const ta = $("#inputText");
  ta.addEventListener("input", () => {
    const n = ta.value.length;
    $("#charCount").textContent = `${n} / ${state.maxChars}`;
    $("#charCount").classList.toggle("over", n >= state.maxChars);
    if (ta.getAttribute("aria-invalid") === "true") setInputError(null);
  });
  ta.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); $("#rewriteForm").requestSubmit(); }
  });
  $$(".segmented[aria-labelledby='decLabel'] button").forEach((b) => b.addEventListener("click", () => {
    state.decoding = b.dataset.dec;
    $$(".segmented[aria-labelledby='decLabel'] button").forEach((x) => x.setAttribute("aria-checked", String(x === b)));
    $("#beamCtl").hidden = state.decoding !== "beam";
    $("#tempCtl").hidden = state.decoding !== "sample";
  }));
  $("#beamSize").addEventListener("input", (e) => ($("#beamOut").textContent = e.target.value));
  $("#temp").addEventListener("input", (e) => ($("#tempOut").textContent = Number(e.target.value).toFixed(1)));
  $("#clearBtn").addEventListener("click", () => {
    ta.value = ""; ta.dispatchEvent(new Event("input")); setInputError(null);
    $("#result").hidden = true; $("#emptyState").hidden = false; $("#insideCard").hidden = true; ta.focus();
  });
  $("#copyBtn").addEventListener("click", async () => {
    try { await navigator.clipboard.writeText($("#politeText").textContent); toast("Copied to the clipboard."); }
    catch { toast("Your browser blocked the clipboard. Select the text and copy it manually.", true); }
  });
  $("#rewriteForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const err = clientCheck(ta.value);
    if (err) { setInputError(err); ta.focus(); return; }
    setInputError(null);
    const btn = $("#goBtn"); btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span><span>Rewriting…</span>';
    try {
      const r = await api("/api/rewrite", {
        text: ta.value, model: $("#modelSelect").value || null, decoding: state.decoding,
        beam_size: +$("#beamSize").value, temperature: +$("#temp").value,
      });
      renderResult(r);
    } catch (ex) {
      setInputError(ex.message);
    } finally {
      btn.disabled = false; btn.innerHTML = "<span>Make it polite</span><kbd>Ctrl ⏎</kbd>";
    }
  });
}

function renderResult(r) {
  $("#emptyState").hidden = true; $("#result").hidden = false;
  // output words; with the copy model, words mostly copied from the input are underlined
  const words = r.output || "(empty output)";
  $("#politeText").innerHTML = r.output_tokens.length
    ? splitDisplay(r.output, r.output_tokens).map(({ text, i }) => {
        const copied = r.p_gen && i != null && r.p_gen[i] < 0.5;
        return i == null ? esc(text) : `<span class="w${copied ? " copied" : ""}" data-i="${i}">${esc(text)}</span>`;
      }).join("")
    : esc(words);
  const label = (state.info.models.find((m) => m.id === r.model) || {}).label || r.model;
  const dec = r.decoding === "beam" ? `beam ${$("#beamSize").value}` : r.decoding === "sample" ? `sampling T=${$("#temp").value}` : "greedy";
  $("#resultMeta").textContent = `${label} · ${dec} · ${r.latency_ms} ms` + (r.p_gen ? " · underlined = copied from your sentence" : "");

  setMeter("pIn", r.politeness_in); setMeter("pOut", r.politeness_out);
  $("#contentV").textContent = r.key_words.length ? `${r.kept.length} / ${r.key_words.length}` : "—";
  $("#keptChips").innerHTML = r.key_words.length
    ? r.key_words.map((w) => `<span class="chip ${r.kept.includes(w) ? "kept" : "lost"}" title="${r.kept.includes(w) ? "kept" : "missing from the output"}">${esc(w)}</span>`).join("")
    : '<span class="muted small">No content words to check in this sentence.</span>';
  const w = $("#warnings");
  w.innerHTML = r.warnings.map((m) => `<div class="warning">${esc(m)}</div>`).join("");
  w.hidden = !r.warnings.length;
  renderInside(r);
}

/* Map each output token to its piece of the detokenised sentence so words can be highlighted. */
function splitDisplay(text, tokens) {
  const out = []; let pos = 0; const low = text.toLowerCase();
  tokens.forEach((t, i) => {
    const at = low.indexOf(t.toLowerCase(), pos);
    if (at < 0) return;
    if (at > pos) out.push({ text: text.slice(pos, at), i: null });
    out.push({ text: text.slice(at, at + t.length), i });
    pos = at + t.length;
  });
  if (pos < text.length) out.push({ text: text.slice(pos), i: null });
  return out;
}

function setMeter(id, v) {
  requestAnimationFrame(() => ($(`#${id}`).style.width = `${Math.max(2, 100 * v)}%`));
  $(`#${id}V`).textContent = pct(v);
}

function renderInside(r) {
  const card = $("#insideCard");
  if (!r.attention || !r.attention.length) {
    card.hidden = false;
    $("#heatmap").classList.add("note");
    $("#heatmap").innerHTML = '<p class="muted">Model A has no attention: the decoder only sees one summary vector of the whole sentence, so there is no map to draw. Pick an attention model to see one.</p>';
  } else {
    card.hidden = false;
    const src = [...r.input_tokens, "<eos>"].slice(0, r.attention[0].length);
    const unk = new Set(r.unknown_words);
    const hm = $("#heatmap");
    hm.classList.remove("note");
    hm.style.gridTemplateColumns = `auto repeat(${src.length}, 26px)`;
    let html = "<span></span>" + src.map((t) => `<span class="col-label${unk.has(t) ? " unk" : ""}">${esc(t)}</span>`).join("");
    r.attention.forEach((row, i) => {
      const t = r.output_tokens[i] ?? "<eos>";
      html += `<span class="row-label">${esc(t)}</span>`;
      row.forEach((a, j) => {
        html += `<span class="cell" data-i="${i}" data-j="${j}" style="background:rgba(var(--heat),${Math.max(0.04, Math.sqrt(a)).toFixed(3)})" data-tip="${esc(t)} ← ${esc(src[j])}: ${(100 * a).toFixed(1)}%"></span>`;
      });
    });
    hm.innerHTML = html;
  }
  // confidence of each generated word (greedy / sampling only)
  const tb = $("#tokenBars");
  if (r.token_probs) {
    tb.innerHTML = '<p class="muted small" style="width:100%;margin:0 0 4px">Confidence of each generated word (probability the model gave it):</p>' +
      r.output_tokens.map((t, i) => `<span class="tb" title="${pct(r.token_probs[i], 1)}"><span class="t">${esc(t)}</span><span class="p"><i style="width:${100 * r.token_probs[i]}%"></i></span></span>`).join("");
  } else tb.innerHTML = '<p class="muted small">Beam search scores whole sentences, so there is no per-word confidence to show.</p>';
}

function initTooltip() {
  const tip = $("#tooltip");
  $("#heatmap").addEventListener("mousemove", (e) => {
    const c = e.target.closest(".cell");
    $$(".polite-text .w.hl").forEach((x) => x.classList.remove("hl"));
    if (!c) { tip.hidden = true; return; }
    tip.textContent = c.dataset.tip; tip.hidden = false;
    tip.style.left = `${Math.min(e.clientX + 14, innerWidth - tip.offsetWidth - 8)}px`;
    tip.style.top = `${e.clientY + 14}px`;
    const w = $(`.polite-text .w[data-i="${c.dataset.i}"]`); if (w) w.classList.add("hl");
  });
  $("#heatmap").addEventListener("mouseleave", () => { tip.hidden = true; $$(".polite-text .w.hl").forEach((x) => x.classList.remove("hl")); });
}

/* ------------------------------------------------------------------ compare tab */
function initCompare() {
  const input = $("#compareText");
  input.addEventListener("input", () => setInputError(null, input, $("#compareError")));
  $("#compareForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const err = clientCheck(input.value);
    if (err) { setInputError(err, input, $("#compareError")); return; }
    const grid = $("#compareGrid");
    grid.innerHTML = '<div class="sys"><div class="skeleton"></div><div class="skeleton" style="width:60%"></div></div>'.repeat(3);
    try {
      const r = await api("/api/compare", { text: input.value });
      const best = Math.max(...r.systems.map(score));
      grid.innerHTML = r.systems.map((s) => {
        const base = s.id.startsWith("baseline");
        const ok = s.politeness_out >= 0.5 && s.content_kept === 1 && !s.repetition;
        return `<article class="sys${base ? " baseline" : ""}${!base && score(s) === best && ok ? " best" : ""}">
          <div class="sys-head"><span class="sys-name">${esc(s.label)}</span>
            <span class="tag ${ok ? "ok" : "no"}">${ok ? "success" : base ? "baseline" : "not quite"}</span></div>
          <p class="sys-out">${esc(s.output)}</p>
          <div class="sys-stats"><span>polite <b>${pct(s.politeness_out)}</b></span>
            <span>kept <b>${s.kept.length}/${s.key_words.length}</b></span>
            ${s.missing.length ? `<span>missing <b>${esc(s.missing.join(", "))}</b></span>` : ""}
            ${s.repetition ? "<span><b>repeats a word</b></span>" : ""}</div></article>`;
      }).join("");
    } catch (ex) {
      grid.innerHTML = ""; setInputError(ex.message, input, $("#compareError"));
    }
  });
}
const score = (s) => s.politeness_out + s.content_kept - (s.repetition ? 1 : 0);

/* ------------------------------------------------------------------ SVG chart helpers */
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

function hbars(el, items, { max = 1, fmt = pct, highlight } = {}) {
  // items: [{label, value, muted}]
  // the viewBox follows the real width of the container so text keeps its real size
  const W = Math.max(300, Math.round(el.clientWidth || 640)), rowH = 30, right = 52;
  const left = Math.min(W * 0.45, 14 + 7 * Math.max(...items.map((d) => d.label.length))), H = items.length * rowH + 10;
  const x = (v) => left + (W - left - right) * (v / max);
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="bar chart">`;
  [0.25, 0.5, 0.75, 1].forEach((t) => (s += `<line class="grid" x1="${x(t * max)}" x2="${x(t * max)}" y1="0" y2="${H - 6}"/>`));
  items.forEach((d, i) => {
    const y = i * rowH + 6;
    const color = d.muted ? css("--ink-3") : d.label === highlight ? css("--accent") : css("--accent");
    s += `<text x="${left - 10}" y="${y + 15}" text-anchor="end">${esc(d.label)}</text>
      <rect x="${left}" y="${y + 3}" width="${Math.max(2, x(d.value || 0) - left)}" height="18" rx="5" fill="${color}" opacity="${d.muted ? 0.45 : d.label === highlight ? 1 : 0.7}"/>
      <text class="val" x="${x(d.value || 0) + 6}" y="${y + 16}">${fmt(d.value)}</text>`;
  });
  el.innerHTML = s + "</svg>";
}

function lines(el, series, { ylabel = "loss" } = {}) {
  // series: [{name, color, points:[[x,y]...], dashed}]
  const pts = series.flatMap((s) => s.points);
  if (!pts.length) { el.innerHTML = '<p class="muted">No training history yet.</p>'; return; }
  const W = 560, H = 280, L = 48, R = 16, T = 12, B = 40;
  const xmax = Math.max(...pts.map((p) => p[0])), ymax = Math.max(...pts.map((p) => p[1])) * 1.05;
  const ymin = Math.min(0, Math.min(...pts.map((p) => p[1])));
  const X = (v) => L + (W - L - R) * ((v - 1) / Math.max(1, xmax - 1));
  const Y = (v) => T + (H - T - B) * (1 - (v - ymin) / (ymax - ymin));
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="line chart">`;
  for (let i = 0; i <= 4; i++) {
    const v = ymin + (ymax - ymin) * (i / 4);
    s += `<line class="grid" x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${L - 6}" y="${Y(v) + 4}" text-anchor="end">${v.toFixed(2)}</text>`;
  }
  const step = Math.max(1, Math.ceil(xmax / 10));
  for (let e = 1; e <= xmax; e += step) s += `<text x="${X(e)}" y="${H - B + 18}" text-anchor="middle">${e}</text>`;
  s += `<text x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">epoch</text>`;
  s += `<text x="12" y="${T + (H - T - B) / 2}" transform="rotate(-90 12 ${T + (H - T - B) / 2})" text-anchor="middle">${ylabel}</text>`;
  series.forEach((sr) => {
    const d = sr.points.map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("");
    s += `<path d="${d}" fill="none" stroke="${sr.color}" stroke-width="2.2" ${sr.dashed ? 'stroke-dasharray="5 4"' : ""} stroke-linejoin="round"/>`;
    const last = sr.points[sr.points.length - 1];
    s += `<circle cx="${X(last[0])}" cy="${Y(last[1])}" r="3.5" fill="${sr.color}"><title>${esc(sr.name)}: ${last[1].toFixed(3)} at epoch ${last[0]}</title></circle>`;
  });
  s += "</svg>";
  el.innerHTML = s + `<div class="legend">${series.map((sr) => `<span><i style="background:${sr.color}"></i>${esc(sr.name)}</span>`).join("")}</div>`;
}

/* ------------------------------------------------------------------ dashboard tabs */
async function loadDashboard() {
  $("#successChart").innerHTML = '<div class="skeleton"></div><div class="skeleton"></div><div class="skeleton" style="width:70%"></div>';
  try { state.dash = await api("/api/dashboard"); renderDashboard(); }
  catch (ex) { $("#successChart").innerHTML = `<p class="error">${esc(ex.message)}</p>`; }
}

function sysLabel(id) {
  if (id === "baseline_copy") return "Copy baseline";
  if (id === "baseline_rules") return "Rule-based baseline";
  const [m, dec] = id.split("__");
  return (state.dash.labels[m] || m) + (dec ? ` (${dec.replace("beam", "beam ")})` : "");
}

const SPLIT_NOTES = {
  test: "Test set: new content, phrasings seen in training. Measured once, after every choice was made on the validation set.",
  test_unseen: "Impolite phrasings (templates) that the model never saw during training: measures whether it generalises beyond memorised patterns.",
  challenge: "72 sentences written by hand, independently of the generator: the hardest and most realistic check.",
};

function renderDashboard() {
  const d = state.dash;
  renderResults();
  // learning curves
  const sel = $("#curveModel"); const models = Object.keys(d.curves);
  if (!sel.options.length) {
    sel.innerHTML = models.map((m) => `<option value="${m}">${esc(d.labels[m] || m)}</option>`).join("");
    sel.value = (state.info && state.info.default_model) || models[0];
    sel.addEventListener("change", renderCurve);
  }
  renderCurve();
  lines($("#allCurves"), models.map((m, i) => ({ name: d.labels[m] || m, color: SERIES[i % SERIES.length],
    points: d.curves[m].map((e) => [e.epoch, e.valid_loss]) })), { ylabel: "validation loss" });
  renderDataset();
}

function renderCurve() {
  const h = state.dash.curves[$("#curveModel").value] || [];
  lines($("#curveChart"), [
    { name: "train loss", color: css("--warm"), points: h.map((e) => [e.epoch, e.train_loss]) },
    { name: "validation loss", color: css("--accent"), points: h.map((e) => [e.epoch, e.valid_loss]) },
  ]);
}

function renderResults() {
  const split = state.split;
  $("#splitNote").textContent = SPLIT_NOTES[split];
  // order: baselines, then the models in the order of the report, each followed by its decodings
  const order = Object.keys(state.dash.labels);
  const rank = (id) => id.startsWith("baseline") ? -1 : order.indexOf(id.split("__")[0]);
  const rows = state.dash.results.filter((r) => r.split === split)
    .sort((a, b) => rank(a.system) - rank(b.system) || a.system.localeCompare(b.system));
  if (!rows.length) { $("#successChart").innerHTML = '<p class="muted">No results for this set yet. Run <code>python -m politely.evaluate</code>.</p>'; $("#resultsTable").innerHTML = ""; return; }
  const final = state.dash.final_system;
  const shown = rows.filter((r) => !r.system.includes("__") || r.system === final);
  const bestModel = rows.find((r) => r.system === final);
  hbars($("#successChart"), shown.map((r) => ({ label: sysLabel(r.system) + (r.system === final ? " ★" : ""), value: r.success_rate, muted: r.system.startsWith("baseline") })),
    { highlight: bestModel && sysLabel(bestModel.system) + " ★" });
  renderHuman();
  const cols = [["BLEU", "BLEU", (v) => num(v, 1)], ["chrF", "chrF", (v) => num(v, 1)], ["content_preservation", "Content kept", pct],
    ["polite_rate", "Polite", pct], ["success_rate", "Success", pct], ["repetition_rate", "Repeats", (v) => pct(v, 1)], ["exact_match", "Exact", pct]];
  const top = {}; cols.forEach(([k]) => (top[k] = k === "repetition_rate" ? Math.min(...rows.map((r) => r[k])) : Math.max(...rows.map((r) => r[k]))));
  $("#resultsTable").innerHTML = `<thead><tr><th>System</th>${cols.map((c) => `<th>${c[1]}</th>`).join("")}</tr></thead><tbody>` +
    rows.map((r) => `<tr class="${r.system.startsWith("baseline") ? "baseline" : ""}${bestModel && r.system === bestModel.system ? " best" : ""}"><td>${esc(sysLabel(r.system))}</td>${cols.map(([k, , f]) =>
      `<td class="${r[k] === top[k] && !r.system.startsWith("baseline") ? "top" : ""}">${f(r[k])}</td>`).join("")}</tr>`).join("") + "</tbody>";
}

function renderHuman() {
  const h = state.dash.human || {}; const el = $("#humanTable"); if (!el) return;
  const rows = [["Test (40 sampled)", "C_copy_nodrop__beam5 (test, random 40)"], ["Unseen phrasings (40 sampled)", "C_copy_nodrop__beam5 (test_unseen, random 40)"],
    ["Hand-written (all 72)", "C_copy_nodrop__beam5"], ["Hand-written, model B without copy", "B_attention"]].filter(([, k]) => h[k]);
  el.innerHTML = rows.length ? "<thead><tr><th>Set</th><th>Automatic success</th><th>Acceptable</th><th>Partial</th><th>Failed</th></tr></thead><tbody>" +
    rows.map(([name, k]) => { const v = h[k]; const n = Object.values(v.counts).reduce((a, b) => a + b, 0);
      return `<tr><td>${name}</td><td>${pct(v.automatic_success)}</td><td class="top">${pct(v.acceptable_rate)}</td><td>${pct((v.counts.partial || 0) / n)}</td><td>${pct((v.counts.failed || 0) / n)}</td></tr>`; }).join("") + "</tbody>"
    : "<tbody><tr><td>No human evaluation found.</td></tr></tbody>";
}

function renderDataset() {
  const ds = state.dash.dataset; if (!ds.splits) return;
  const sp = ds.splits; const tr = sp.train;
  const totalPairs = ["train", "valid", "test", "test_unseen"].reduce((a, s) => a + (sp[s] ? sp[s].pairs : 0), 0);
  const kpi = [[totalPairs.toLocaleString("en"), "sentence pairs in total"], [tr.inputs.toLocaleString("en"), "training inputs"],
    [Object.keys(tr.categories).length, "balanced categories"], [(state.dash.vocab.vocab_size || "—").toLocaleString("en"), "vocabulary entries (4 special tokens)"],
    [num(tr.impolite_len_mean, 1), "tokens per impolite input"], [pct(tr.token_overlap_mean), "target tokens already in the input"]];
  $("#kpis").innerHTML = kpi.map(([b, s]) => `<div class="kpi"><b>${b}</b><span>${s}</span></div>`).join("");
  hbars($("#catChart"), Object.entries(tr.categories).map(([c, v]) => ({ label: c, value: v })),
    { max: Math.max(...Object.values(tr.categories)), fmt: (v) => String(v) });
  const names = { train: "Train", valid: "Validation", test: "Test", test_unseen: "Test (unseen phrasings)", challenge: "Challenge (hand-written)" };
  $("#splitTable").innerHTML = "<thead><tr><th>Split</th><th>Inputs</th><th>Pairs</th><th>Refs</th><th>Templates</th></tr></thead><tbody>" +
    Object.entries(sp).map(([s, v]) => `<tr><td>${names[s] || s}</td><td>${v.inputs.toLocaleString("en")}</td><td>${v.pairs.toLocaleString("en")}</td><td>${v.references_per_input}</td><td>${v.impolite_templates ?? "—"}</td></tr>`).join("") + "</tbody>";
  const c = ds.checks || {}; const shared = Object.values(c.inputs_shared_between_splits || {}).reduce((a, b) => a + b, 0);
  $("#checks").innerHTML = [`${shared} input sentences shared between splits`, `${c.duplicate_pairs} duplicate pairs`,
    `${c.pairs_with_identical_input_output} pairs where the output equals the input`, "No word from the offensive-word filter list appears"].map((x) => `<li>${x}</li>`).join("");
  $("#examplesList").innerHTML = (state.dash.examples || []).map((e) =>
    `<div class="ex"><span class="cat">${esc(e.category)}</span><p class="imp">${esc(e.impolite)}</p><p class="pol">${esc(e.polite)}</p></div>`).join("");
}

function initSplitSeg() {
  $$("#splitSeg button").forEach((b) => b.addEventListener("click", () => {
    state.split = b.dataset.split;
    $$("#splitSeg button").forEach((x) => x.setAttribute("aria-checked", String(x === b)));
    if (state.dash) renderResults();
  }));
}

/* ------------------------------------------------------------------ start */
async function init() {
  initTheme(); initTabs(); initRewrite(); initTooltip(); initCompare(); initSplitSeg();
  try {
    state.info = await api("/api/info");
  } catch (ex) {
    setInputError(ex.message); return;
  }
  state.maxChars = state.info.max_chars;
  $("#charCount").textContent = `0 / ${state.maxChars}`;
  const sel = $("#modelSelect");
  if (!state.info.models.length) {
    setInputError("No trained model was found in checkpoints/. Train one first: python -m politely.train --config configs/B_attention.json");
    $("#goBtn").disabled = true;
  }
  sel.innerHTML = state.info.models.map((m) => `<option value="${m.id}" title="${esc(m.description)}">${esc(m.label)}${m.id === state.info.default_model ? " — recommended" : ""}</option>`).join("");
  if (state.info.default_model) sel.value = state.info.default_model;
  // two examples per category at most, shuffled order of categories kept stable
  const chips = state.info.examples.slice(0, 12);
  $("#exampleChips").innerHTML = chips.map((e, i) => `<button type="button" class="chip" data-i="${i}" title="${esc(e.category)}">${esc(e.text)}</button>`).join("");
  $$("#exampleChips .chip").forEach((c) => c.addEventListener("click", () => {
    const ta = $("#inputText"); ta.value = chips[+c.dataset.i].text; ta.dispatchEvent(new Event("input"));
    $("#rewriteForm").requestSubmit();
  }));
  $("#compareText").value = "";
}
let resizeTimer;
addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => state.dash && renderDashboard(), 200); });
document.addEventListener("DOMContentLoaded", init);
