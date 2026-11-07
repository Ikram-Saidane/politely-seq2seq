// The final model (LSTM encoder–decoder + attention + copy mechanism) running in the browser.
// Same computations as model.py / inference.py, with the weights exported by export_model.py.

const PAD = 0, SOS = 1, EOS = 2, UNK = 3;

// ---------------------------------------------------------------------------
// Text processing (same rules as preprocessing.py)
// ---------------------------------------------------------------------------
const QUOTES = { "‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-" };
const TOKEN_RE = /[\p{L}\p{N}_]+(?:'[\p{L}\p{N}_]+)?|[^\p{L}\p{N}_\s]/gu;

function cleanChars(text) {
  return String(text).normalize("NFKC").replace(/[‘’“”–—]/g, (c) => QUOTES[c]);
}

export function normalize(text) {
  return cleanChars(text).toLowerCase().replace(/\s+/gu, " ").trim();
}

export function tokenize(text) {
  return normalize(text).match(TOKEN_RE) || [];
}

export function detokenize(tokens, capitalize = false) {
  let text = tokens.join(" ");
  text = text.replace(/ ([.,!?;:%)\]}])/g, "$1");
  text = text.replace(/([(\[{$]) /g, "$1");
  if (capitalize) {
    text = text.replace(/([\p{L}\p{N}_]) - ([\p{L}\p{N}_])/gu, "$1-$2");
    text = text.replace(/\bi\b/g, "I");
    text = text.replace(/\bi'(m|d|ll|ve)\b/g, (m, s) => "I'" + s);
    text = text.replace(/(^|[.!?]\s+)([a-z])/g, (m, a, b) => a + b.toUpperCase());
  }
  return text;
}

export function restoreCase(tokens, sourceText) {
  // give the output words the case they had in the input (tv -> TV, sarah -> Sarah)
  const cased = new Map();
  let prev = ".";
  for (const t of cleanChars(sourceText).match(TOKEN_RE) || []) {
    const startOfSentence = [".", "!", "?", ":"].includes(prev);
    const special = t.length > 1 && t.slice(1).toLowerCase() !== t.slice(1);
    if (t.toLowerCase() !== t && (special || !startOfSentence) && !cased.has(t.toLowerCase())) {
      cased.set(t.toLowerCase(), t);
    }
    prev = t;
  }
  return tokens.map((t) => cased.get(t) ?? t);
}

export function stem(word) {
  let w = word.toLowerCase();
  if (w.endsWith("'s")) w = w.slice(0, -2);
  for (const suf of ["ing", "ed", "es", "s"]) {
    if (w.endsWith(suf) && w.length - suf.length >= 3) { w = w.slice(0, -suf.length); break; }
  }
  if (w.length >= 4 && w.at(-1) === w.at(-2) && !"lsz".includes(w.at(-1))) w = w.slice(0, -1);
  return w.replace(/e+$/, "") || w;
}

// ---------------------------------------------------------------------------
// Small linear-algebra helpers (Float32 weights, Float64 activations)
// ---------------------------------------------------------------------------
function linear(W, b, x, rows, cols) {
  const y = new Float64Array(rows);
  for (let r = 0; r < rows; r++) {
    let s = b ? b[r] : 0;
    const off = r * cols;
    for (let c = 0; c < cols; c++) s += W[off + c] * x[c];
    y[r] = s;
  }
  return y;
}
const sigmoid = (x) => 1 / (1 + Math.exp(-x));
function concat(...arrays) {
  const out = new Float64Array(arrays.reduce((n, a) => n + a.length, 0));
  let o = 0;
  for (const a of arrays) { out.set(a, o); o += a.length; }
  return out;
}

// ---------------------------------------------------------------------------
// The model
// ---------------------------------------------------------------------------
export class PolitelyModel {
  static async load(base = "model/") {
    const [info, buffer] = await Promise.all([
      fetch(base + "model.json").then((r) => r.json()),
      fetch(base + "weights.bin").then((r) => r.arrayBuffer()),
    ]);
    return new PolitelyModel(info, buffer);
  }

  constructor(info, buffer) {
    const all = new Float32Array(buffer);
    this.t = {};
    for (const [name, { offset, shape }] of Object.entries(info.tensors)) {
      this.t[name] = all.subarray(offset, offset + shape.reduce((a, b) => a * b, 1));
    }
    this.cfg = info.config;
    this.itos = info.vocab;
    this.stoi = new Map(this.itos.map((w, i) => [w, i]));
    this.V = this.itos.length;
    this.E = this.cfg.emb_dim;
    this.H = this.cfg.hid_dim;
    this.A = this.cfg.attn_dim;
    this.stop = new Set(info.stop);
    this.punct = new Set(info.punct);
    this.rude = new Set(info.rude_words);
    this.clf = info.politeness;
  }

  embed(id) {
    return Float64Array.from(this.t["embedding.weight"].subarray(id * this.E, (id + 1) * this.E));
  }

  lstm(prefix, x, h, c) {
    // PyTorch gate order: input, forget, cell, output
    const H = this.H, t = this.t;
    const gi = linear(t[prefix + "weight_ih_l0"], t[prefix + "bias_ih_l0"], x, 4 * H, x.length);
    const gh = linear(t[prefix + "weight_hh_l0"], t[prefix + "bias_hh_l0"], h, 4 * H, H);
    const h2 = new Float64Array(H), c2 = new Float64Array(H);
    for (let j = 0; j < H; j++) {
      const i = sigmoid(gi[j] + gh[j]);
      const f = sigmoid(gi[H + j] + gh[H + j]);
      const g = Math.tanh(gi[2 * H + j] + gh[2 * H + j]);
      const o = sigmoid(gi[3 * H + j] + gh[3 * H + j]);
      c2[j] = f * c[j] + i * g;
      h2[j] = o * Math.tanh(c2[j]);
    }
    return { h: h2, c: c2 };
  }

  encode(tokens) {
    // ids, and "extended" ids so that unknown input words can be copied
    const oovs = [], srcExt = [];
    const ids = tokens.map((w) => this.stoi.get(w) ?? UNK);
    for (const w of tokens) {
      if (this.stoi.has(w)) srcExt.push(this.stoi.get(w));
      else {
        if (!oovs.includes(w)) oovs.push(w);
        srcExt.push(this.V + oovs.indexOf(w));
      }
    }
    let state = { h: new Float64Array(this.H), c: new Float64Array(this.H) };
    const encOut = [];
    for (const id of ids) {
      state = this.lstm("encoder.rnn.", this.embed(id), state.h, state.c);
      encOut.push(state.h);
    }
    const keys = encOut.map((h) => linear(this.t["decoder.attention.W_enc.weight"], null, h, this.A, this.H));
    return { ids, srcExt, oovs, encOut, keys, state };
  }

  step(inp, state, enc) {
    // one decoding step: previous word -> log-probabilities of the next word (+ copies)
    const t = this.t, H = this.H, A = this.A;
    const emb = this.embed(inp);
    // Bahdanau attention
    const wd = linear(t["decoder.attention.W_dec.weight"], t["decoder.attention.W_dec.bias"], state.h, A, H);
    const v = t["decoder.attention.v.weight"];
    const scores = enc.keys.map((k) => {
      let s = 0;
      for (let a = 0; a < A; a++) s += v[a] * Math.tanh(k[a] + wd[a]);
      return s;
    });
    const mx = Math.max(...scores);
    const ex = scores.map((s) => Math.exp(s - mx));
    const tot = ex.reduce((a, b) => a + b, 0);
    const weights = ex.map((e) => e / tot);
    const context = new Float64Array(H);
    enc.encOut.forEach((h, j) => { for (let d = 0; d < H; d++) context[d] += weights[j] * h[d]; });
    // decoder LSTM and output layer
    const next = this.lstm("decoder.rnn.", concat(emb, context), state.h, state.c);
    const s = next.h;
    const pre = linear(t["decoder.pre_out.weight"], t["decoder.pre_out.bias"], concat(s, emb, context), this.E, 2 * H + this.E);
    const logits = linear(t["decoder.out.weight"], t["decoder.out.bias"], pre, this.V, this.E);
    const lmx = Math.max(...logits);
    let z = 0;
    for (let i = 0; i < this.V; i++) z += Math.exp(logits[i] - lmx);
    // copy mechanism: P(w) = p_gen * P_vocab(w) + (1 - p_gen) * attention on w
    const pgen = sigmoid(linear(t["decoder.p_gen.weight"], t["decoder.p_gen.bias"], concat(context, s, emb), 1, 2 * H + this.E)[0]);
    const dist = new Float64Array(this.V + enc.oovs.length);
    for (let i = 0; i < this.V; i++) dist[i] = pgen * Math.exp(logits[i] - lmx) / z;
    enc.srcExt.forEach((id, j) => { dist[id] += (1 - pgen) * weights[j]; });
    const logp = dist.map((p) => Math.log(p + 1e-12));
    return { logp, state: next, weights, pgen };
  }

  toInput(id) {
    return id >= this.V ? UNK : id;
  }

  decodeIds(ids, oovs) {
    const out = [];
    for (const i of ids) {
      if (i === EOS) break;
      if (i === PAD || i === SOS) continue;
      if (i < this.V) out.push(this.itos[i]);
      else out.push(oovs[i - this.V] ?? "<UNK>");
    }
    return out;
  }

  greedy(tokens) {
    const enc = this.encode(tokens);
    let state = enc.state, inp = SOS;
    const ids = [], attention = [], pgens = [];
    for (let n = 0; n < this.cfg.max_decode_len; n++) {
      const r = this.step(inp, state, enc);
      let best = 0;
      for (let i = 1; i < r.logp.length; i++) if (r.logp[i] > r.logp[best]) best = i;
      ids.push(best);
      attention.push(r.weights);
      pgens.push(r.pgen);
      if (best === EOS) break;
      state = r.state;
      inp = this.toInput(best);
    }
    return { tokens: this.decodeIds(ids, enc.oovs), attention, pgen: pgens };
  }

  beam(tokens, k = 5, alpha = 1.0) {
    const enc = this.encode(tokens);
    let beams = [{ score: 0, seq: [SOS], state: enc.state, att: [], pg: [] }];
    let finished = [];
    for (let n = 0; n < this.cfg.max_decode_len; n++) {
      const steps = beams.map((b) => this.step(this.toInput(b.seq.at(-1)), b.state, enc));
      const candidates = [];
      steps.forEach((r, i) => {
        // the k most probable next words of this hypothesis
        const order = Array.from(r.logp.keys()).sort((a, b) => r.logp[b] - r.logp[a] || a - b).slice(0, k);
        for (const id of order) candidates.push({ score: beams[i].score + r.logp[id], seq: [...beams[i].seq, id], i });
      });
      candidates.sort((a, b) => b.score - a.score);
      const next = [];
      for (const c of candidates) {
        const b = beams[c.i], r = steps[c.i];
        const att = [...b.att, r.weights], pg = [...b.pg, r.pgen];
        if (c.seq.at(-1) === EOS) {
          // length-normalised score, otherwise short sentences always win
          finished.push({ score: c.score / Math.pow(c.seq.length - 1, alpha), seq: c.seq, att, pg });
        } else {
          next.push({ score: c.score, seq: c.seq, state: r.state, att, pg });
        }
        if (next.length === k) break;
      }
      beams = next;
      if (finished.length >= k || beams.length === 0) break;
    }
    if (finished.length === 0) {
      finished = beams.map((b) => ({ ...b, score: b.score / Math.pow(b.seq.length - 1, alpha) }));
    }
    let best = finished[0];
    for (const f of finished) if (f.score > best.score) best = f;
    return { tokens: this.decodeIds(best.seq.slice(1), enc.oovs), attention: best.att, pgen: best.pg };
  }

  // -------------------------------------------------------------------------
  // Politeness classifier and content words (same as politeness.py / app.py)
  // -------------------------------------------------------------------------
  politeness(tokens) {
    const toks = ["<s>", ...tokens, "</s>"];
    const feats = new Set(tokens);
    for (let i = 0; i + 1 < toks.length; i++) feats.add(toks[i] + " " + toks[i + 1]);
    let s = this.clf.b;
    for (const f of feats) {
      const j = this.clf.vocab[f];
      if (j !== undefined) s += this.clf.w[j];
    }
    return sigmoid(s);
  }

  contentWords(tokens, output) {
    const keys = [...new Set(tokens)].filter((w) => !this.stop.has(w) && !this.punct.has(w) && !this.rude.has(w));
    const outStems = new Set(output.map(stem));
    const kept = keys.filter((w) => outStems.has(stem(w)));
    return { keys, kept, lost: keys.filter((w) => !kept.includes(w)) };
  }

  rewrite(text, strategy = "beam") {
    const tokens = tokenize(text);
    const r = strategy === "beam" ? this.beam(tokens, 5) : this.greedy(tokens);
    const content = this.contentWords(tokens, r.tokens);
    return {
      inputTokens: tokens,
      tokens: r.tokens,
      text: detokenize(restoreCase(r.tokens, text), true),
      attention: r.attention,
      pgen: r.pgen,
      unknown: [...new Set(tokens)].filter((w) => !this.stoi.has(w) && !this.punct.has(w)),
      politenessBefore: this.politeness(tokens),
      politenessAfter: r.tokens.length ? this.politeness(r.tokens) : 0,
      ...content,
    };
  }
}
