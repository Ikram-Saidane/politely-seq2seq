import { PolitelyModel, tokenize, stem } from "./model.js";

// Sentences written by hand (challenge set) and a few new ones
const EXAMPLES = [
  "Give me the keys to the storeroom right now.",
  "Turn the music down.",
  "You're wrong. Water boils at 100 degrees.",
  "My room has no hot water.",
  "Stop interrupting me during meetings.",
  "I won't lend you any more money.",
];

// Success rates from reports/results.json (C = final system, beam search 5)
const RESULTS = [
  ["Copy the input", "baseline", 0.000, 0.034, 0.000],
  ["Rules: remove rude words, add “please”", "baseline", 0.125, 0.253, 0.097],
  ["A · LSTM encoder–decoder", "no attention", 0.845, 0.350, 0.111],
  ["B · + attention", "", 0.912, 0.582, 0.222],
  ["C · + copy mechanism", "final model, beam search", 0.999, 0.787, 0.764],
];

// Real outputs of the final model on hand-written sentences (reports/human_eval_challenge_*.csv)
const PAIRS = [
  ["Turn the music down.", "Could I ask you to turn the music down? Thank you!", true, "Acceptable"],
  ["You're wrong. Water boils at 100 degrees.", "Actually, I think water boils at 100 degrees. Could you double-check?", true, "Acceptable"],
  ["I won't lend you any more money.", "Would it be possible for you to lend you any more money?", false, "Failed: the refusal became a request"],
  ["You were supposed to pay the rent yesterday.", "Whenever you get a chance, could you pay the rent yesterday?", false, "Partial: the tense is wrong"],
];

const $ = (id) => document.getElementById(id);
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const pct = (x) => `${Math.round(x * 100)}%`;

// ---------- static sections ----------
$("results-body").innerHTML = RESULTS.map(([name, note, ...v], i) => `
  <tr class="${i === RESULTS.length - 1 ? "final" : ""}">
    <td class="system">${esc(name)}${note ? `<span>${esc(note)}</span>` : ""}</td>
    ${v.map((x) => `<td><div class="bar"><b>${x.toFixed(3)}</b><i style="width:${Math.max(x * 110, 2)}px" aria-hidden="true"></i></div></td>`).join("")}
  </tr>`).join("");

$("pairs").innerHTML = PAIRS.map(([inp, out, ok, verdict]) => `
  <div class="pair ${ok ? "good" : "bad"}">
    <p class="in">${esc(inp)}</p>
    <p class="out">${esc(out)}</p>
    <p class="verdict">${ok ? "✓" : "✗"} ${esc(verdict)}</p>
  </div>`).join("");

const exBox = document.querySelector(".examples");
for (const ex of EXAMPLES) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "chip";
  b.textContent = ex;
  b.addEventListener("click", () => { $("input").value = ex; run(); });
  exBox.appendChild(b);
}
for (const label of document.querySelectorAll(".seg label")) {
  const input = label.querySelector("input");
  const span = document.createElement("span");
  span.textContent = label.textContent.trim();
  label.textContent = "";
  label.append(input, span);
}

// human evaluation of the 72 hand-written sentences (reports/human_eval_summary.json)
const HUMAN = [["B · attention, no copy", 12, 4, 56], ["C · + copy (final)", 38, 12, 22]];
$("human").innerHTML = HUMAN.map(([name, a, p, f]) => {
  const n = a + p + f;
  return `<div class="hrow"><span class="hname">${esc(name)}</span><div class="hbar">
    <i class="acc" style="flex:${a}" title="acceptable: ${a}">${pct(a / n)}</i><i class="par" style="flex:${p}" title="partial: ${p}">${p >= 8 ? pct(p / n) : ""}</i><i class="fail" style="flex:${f}" title="failed: ${f}">${pct(f / n)}</i></div></div>`;
}).join("") + `<p class="legend"><span class="acc"></span>acceptable <span class="par"></span>partial <span class="fail"></span>failed</p>`;

// learning curves (experiments/*/history.json)
fetch("model/curves.json").then((r) => r.json()).then((curves) => {
  const series = [["A_lstm", "A · no attention", "#9AA3B5"], ["B_attention", "B · attention", "#E0A33A"], ["C_copy_nodrop", "C · copy (final)", "#127A68"]];
  const W = 560, H = 300, L = 44, B = 46, T = 12, R = 12;
  const maxEpoch = Math.max(...Object.values(curves).map((c) => c.length));
  const x = (e) => L + (e - 1) / (maxEpoch - 1) * (W - L - R);
  const yMax = 1.2, y = (v) => T + (1 - Math.min(v, yMax) / yMax) * (H - T - B);
  let svg = `<svg viewBox="0 0 ${W} ${H}" width="100%">`;
  for (const v of [0, 0.3, 0.6, 0.9, 1.2]) {
    svg += `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" class="grid"/><text x="${L - 8}" y="${y(v) + 4}" text-anchor="end">${v.toFixed(1)}</text>`;
  }
  for (let e = 5; e <= maxEpoch; e += 5) svg += `<text x="${x(e)}" y="${H - B + 18}" text-anchor="middle">${e}</text>`;
  svg += `<text x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">epoch</text>`;
  for (const [key, label, color] of series) {
    const c = curves[key];
    const path = (k) => c.map((p, i) => `${i ? "L" : "M"}${x(p.epoch).toFixed(1)},${y(p[k]).toFixed(1)}`).join("");
    svg += `<path d="${path("train")}" fill="none" stroke="${color}" stroke-width="1.5" stroke-dasharray="4 4" opacity="0.8"/>`;
    svg += `<path d="${path("valid")}" fill="none" stroke="${color}" stroke-width="2.5"/>`;
  }
  $("curves").innerHTML = `<p class="legend">${series.map(([, label, color]) =>
    `<span style="background:${color}"></span>${esc(label)}`).join(" ")}</p>` + svg + "</svg>";
});

// ---------- the demo ----------
let model = null, modelB = null, lastText = "", lastPolite = "";

function highlight(text, words) {
  // marks the words of `text` whose stem is one of the content words
  const stems = new Set(words.map(stem));
  return text.replace(/[\p{L}\p{N}_]+(?:'[\p{L}\p{N}_]+)?/gu, (w) =>
    stems.has(stem(w.toLowerCase())) ? `<mark>${esc(w)}</mark>` : esc(w));
}

function heatmap(inputTokens, outputTokens, attention) {
  const rows = [...outputTokens, "<end>"].slice(0, attention.length);
  const cell = 26, left = 110, top = 70;
  const w = left + inputTokens.length * cell + 10, h = top + rows.length * cell + 10;
  let svg = `<svg width="${w}" height="${h}" role="img" aria-label="Attention map">`;
  inputTokens.forEach((t, j) => {
    svg += `<text transform="translate(${left + j * cell + cell / 2 + 4},${top - 8}) rotate(-50)">${esc(t)}</text>`;
  });
  rows.forEach((t, i) => {
    svg += `<text x="${left - 8}" y="${top + i * cell + cell / 2 + 4}" text-anchor="end">${esc(t)}</text>`;
    attention[i].forEach((a, j) => {
      svg += `<rect x="${left + j * cell}" y="${top + i * cell}" width="${cell - 2}" height="${cell - 2}" rx="3" fill="rgb(18,122,104)" fill-opacity="${(0.06 + 0.94 * a).toFixed(3)}"><title>${esc(t)} ← ${esc(inputTokens[j])}: ${a.toFixed(2)}</title></rect>`;
    });
  });
  return svg + "</svg>";
}

function run() {
  if (!model) return;
  const text = $("input").value.trim();
  $("error").hidden = true;
  if (!text || !/\p{L}/u.test(text)) return showError("Type a sentence with words first.");
  const tokens = tokenize(text);
  if (tokens.length > 40) return showError(`The model handles up to 40 words and punctuation marks; your sentence has ${tokens.length}.`);
  const strategy = document.querySelector('input[name="decoding"]:checked').value;
  $("go").disabled = true;
  $("go").textContent = "Rewriting…";
  setTimeout(() => {
    const t0 = performance.now();
    const r = model.rewrite(text, strategy);
    const ms = performance.now() - t0;
    $("before").innerHTML = highlight(text, r.keys);
    $("after").innerHTML = r.text ? highlight(r.text, r.kept) : "(no output)";
    $("pb").textContent = pct(r.politenessBefore);
    $("pa").textContent = pct(r.politenessAfter);
    $("pa").className = r.politenessAfter > r.politenessBefore ? "up" : "";
    $("ck").textContent = r.keys.length ? `${r.kept.length} of ${r.keys.length}` : "—";
    $("tm").textContent = `${Math.round(ms)} ms`;
    const notes = [];
    if (r.lost.length) notes.push(`<span class="lost">Lost: ${r.lost.map(esc).join(", ")}.</span>`);
    if (r.unknown.length) notes.push(`Words never seen in training: ${r.unknown.map(esc).join(", ")}. The copy mechanism can still reuse them.`);
    if (r.politenessAfter < 0.5) notes.push("The politeness classifier does not find the rewrite polite enough.");
    $("notes").innerHTML = notes.join(" ");
    $("heatmap").innerHTML = heatmap(r.inputTokens, r.tokens, r.attention);
    $("in-tokens").innerHTML = r.inputTokens.map((t, i) =>
      `<span class="tok"><b>${esc(t)}</b><small>${r.inputIds[i] === 3 ? "3 &lt;UNK&gt;" : r.inputIds[i]}</small></span>`).join("") +
      `<span class="tok end"><b>&lt;EOS&gt;</b><small>2</small></span>`;
    $("copy-bars").innerHTML = [...r.tokens, "<end>"].slice(0, r.pgen.length).map((t, i) => {
      const c = 1 - r.pgen[i];
      return `<div class="cb"><span>${esc(t)}</span><i><b style="width:${(c * 100).toFixed(0)}%"></b></i><em>${c >= 0.5 ? "copied" : "generated"}</em></div>`;
    }).join("");
    lastText = text;
    lastPolite = r.text;
    $("compare-box").hidden = true;
    $("compare").disabled = false;
    $("result").hidden = false;
    $("go").disabled = false;
    $("go").textContent = "Rewrite politely";
  }, 30);
}

function showError(msg) {
  $("error").textContent = msg;
  $("error").hidden = false;
}

$("go").addEventListener("click", run);

$("copy").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(lastPolite);
    $("copy").textContent = "Copied";
  } catch {
    $("copy").textContent = "Select the sentence to copy it";
  }
  setTimeout(() => { $("copy").textContent = "Copy the polite sentence"; }, 1800);
});

$("compare").addEventListener("click", async () => {
  const btn = $("compare");
  btn.disabled = true;
  try {
    if (!modelB) {
      btn.textContent = "Loading model B…";
      modelB = await PolitelyModel.load("model/B/", (f) => { btn.textContent = `Loading model B… ${Math.round(f * 100)}%`; });
    }
    const strategy = document.querySelector('input[name="decoding"]:checked').value;
    const r = modelB.rewrite(lastText, strategy);
    $("b-out").innerHTML = r.text ? highlight(r.text, r.kept) : "(no output)";
    $("b-note").textContent = `Same encoder–decoder with attention but without the copy mechanism: ${r.kept.length} of ${r.keys.length} content words kept` +
      (r.lost.length ? ` (lost: ${r.lost.join(", ")}).` : ".");
    $("compare-box").hidden = false;
  } catch {
    $("b-note").textContent = "Model B could not be loaded. Reload the page to try again.";
    $("compare-box").hidden = false;
  }
  btn.textContent = "Compare with model B (no copy mechanism)";
});
$("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); run(); }
});

PolitelyModel.load("model/", (f) => {
  $("progress-bar").style.width = `${Math.round(f * 100)}%`;
  $("go").textContent = `Loading the model… ${Math.round(f * 100)}%`;
}).then((m) => {
  model = m;
  $("progress").classList.add("done");
  $("go").disabled = false;
  $("go").textContent = "Rewrite politely";
  run();
}).catch(() => {
  $("go").textContent = "Model unavailable";
  showError("The model could not be loaded. Reload the page to try again.");
});
