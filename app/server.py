"""Politely — web interface (FastAPI back end + static HTML/CSS/JS front end).

    python -m uvicorn app.server:app --port 8000        (from the project root)
    or simply: python run_app.py

Endpoints
    GET  /                 the single-page interface
    GET  /api/info         available models, decoding strategies, example sentences
    POST /api/rewrite      rewrite one sentence with one model (+ attention map, scores, warnings)
    POST /api/compare      rewrite one sentence with every model and both baselines
    GET  /api/dashboard    dataset statistics, test results and learning curves for the charts
"""
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from politely.baselines import RUDE_MARKERS, copy_baseline, rules_baseline  # noqa: E402
from politely.decode import beam_search, greedy_decode, load_model  # noqa: E402
from politely.metrics import PUNCT, STOP, has_repetition, stem  # noqa: E402
from politely.politeness import PolitenessClassifier  # noqa: E402
from politely.preprocessing import detokenize, restore_case, tokenize  # noqa: E402

MAX_CHARS, MAX_TOKENS = 300, 40
MODEL_INFO = {
    "A_lstm": ("A · LSTM encoder–decoder", "Baseline: the whole sentence is squeezed into one vector."),
    "B_attention": ("B · + Bahdanau attention", "The decoder looks back at every input word at each step."),
    "C_copy_nodrop": ("C · + copy mechanism", "Can copy input words (names, numbers, unknown words) instead of generating them."),
    "C_copy": ("C' · copy + word dropout", "Model C trained with 10% of input words hidden behind <UNK> (ablation)."),
    "D_gru_attention": ("D · GRU + attention", "Model B with GRU cells instead of LSTM cells."),
    "E_attention_h128": ("E · attention, hidden 128", "Model B with a smaller hidden state."),
    "E_attention_h512": ("E · attention, hidden 512", "Model B with a larger hidden state."),
    "F_attention_tf0.5": ("F · attention, teacher forcing 0.5", "Model B trained on its own predictions half of the time."),
}
FINAL_SYSTEM = "C_copy_nodrop__beam5"   # chosen on the validation set (report, section 7.3)
RUDE_WORDS = {w for m in RUDE_MARKERS for w in m.split()}

app = FastAPI(title="Politely", docs_url="/api/docs", redoc_url=None)
app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")


# --------------------------------------------------------------------------------------- resources
def available_models():
    return [m for m in MODEL_INFO if (ROOT / "checkpoints" / f"{m}.pt").exists()]


@lru_cache(maxsize=None)
def get_model(name):
    return load_model(name, "cpu")


@lru_cache(maxsize=1)
def get_clf():
    return PolitenessClassifier.load()


def read_json(path, default=None):
    p = ROOT / path
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


@lru_cache(maxsize=1)
def default_model():
    """The model with the best success rate on the VALIDATION split (greedy decoding)."""
    best, score = None, -1
    for m in available_models():
        r = read_json(f"experiments/{m}/valid_metrics.json")
        if r and r["success_rate"] > score:
            best, score = m, r["success_rate"]
    return best or (available_models() or [None])[0]


# --------------------------------------------------------------------------------------- helpers
class RewriteRequest(BaseModel):
    text: str = Field(..., description="the impolite sentence")
    model: str | None = None
    decoding: str = "greedy"          # greedy | beam | sample
    beam_size: int = Field(5, ge=1, le=10)
    temperature: float = Field(0.8, gt=0.0, le=2.0)


def validate(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        raise HTTPException(422, "Please type a sentence to rewrite.")
    if len(text) > MAX_CHARS:
        raise HTTPException(422, f"Please enter one sentence of at most {MAX_CHARS} characters "
                                 f"(you typed {len(text)}).")
    if not any(c.isalpha() for c in text):
        raise HTTPException(422, "The sentence must contain words, not only numbers or symbols.")
    tokens = tokenize(text)
    if len(tokens) > MAX_TOKENS:
        raise HTTPException(422, f"The model handles sentences of up to {MAX_TOKENS} words and "
                                 f"punctuation marks; yours has {len(tokens)}. Please shorten it.")
    return tokens


def key_words(tokens):
    """Words that carry the content of the input (what a faithful rewrite should keep)."""
    return [t for t in dict.fromkeys(tokens) if t not in STOP and t not in PUNCT and t not in RUDE_WORDS]


def describe(tokens, out_tokens):
    keys = key_words(tokens)
    out_stems = {stem(t) for t in out_tokens}
    kept = [k for k in keys if stem(k) in out_stems]       # "close" is kept by "closing"
    p_in, p_out = get_clf().predict_proba([tokens, out_tokens]) if out_tokens else (get_clf().predict_proba([tokens])[0], 0.0)
    return {
        "politeness_in": round(float(p_in), 3), "politeness_out": round(float(p_out), 3),
        "key_words": keys, "kept": kept, "missing": [k for k in keys if k not in kept],
        "content_kept": round(len(kept) / len(keys), 3) if keys else 1.0,
        "repetition": has_repetition(out_tokens),
    }


def run_model(name, tokens, decoding="greedy", beam_size=5, temperature=0.8):
    model, vocab = get_model(name)
    if decoding == "beam":
        r = beam_search(model, tokens, vocab, k=beam_size)
        probs = None
    elif decoding == "sample":
        r = greedy_decode(model, [tokens], vocab, sample={"temperature": temperature, "seed": int(time.time())})[0]
        probs = r["token_probs"]
    else:
        r = greedy_decode(model, [tokens], vocab)[0]
        probs = r["token_probs"]
    return r, probs, vocab


# --------------------------------------------------------------------------------------- routes
@app.get("/")
def index():
    return FileResponse(ROOT / "app" / "static" / "index.html")


@app.get("/api/info")
def info():
    models = []
    for m in available_models():
        label, desc = MODEL_INFO[m]
        r = read_json(f"experiments/{m}/test_metrics.json", {})
        s = read_json(f"experiments/{m}/train_summary.json", {})
        models.append({"id": m, "label": label, "description": desc,
                       "parameters": (s.get("params") or {}).get("total"),
                       "test_success": r.get("success_rate"), "test_bleu": r.get("BLEU")})
    examples = []          # in-distribution test sentences first, then a few hand-written ones
    test = ROOT / "data" / "test.csv"
    if test.exists():
        df = pd.read_csv(test).drop_duplicates("impolite")
        for cat, g in df.groupby("category"):
            examples.append({"category": cat, "text": g.sample(1, random_state=7).iloc[0]["impolite"],
                             "source": "test"})
    if (ROOT / "data" / "challenge.csv").exists():
        ch = pd.read_csv(ROOT / "data" / "challenge.csv")
        for cat, g in ch.groupby("category"):
            examples.append({"category": cat, "text": g.iloc[0]["impolite"], "source": "hand-written"})
    return {"models": models, "default_model": default_model(), "max_tokens": MAX_TOKENS,
            "max_chars": MAX_CHARS, "examples": examples}


@app.post("/api/rewrite")
def rewrite(req: RewriteRequest):
    tokens = validate(req.text)
    name = req.model or default_model()
    if name not in available_models():
        raise HTTPException(404, f"Model '{name}' is not available. Train it first or pick another model.")
    if req.decoding not in ("greedy", "beam", "sample"):
        raise HTTPException(422, "Decoding must be 'greedy', 'beam' or 'sample'.")
    t0 = time.time()
    r, probs, vocab = run_model(name, tokens, req.decoding, req.beam_size, req.temperature)
    out = r["tokens"]
    unknown = [t for t in dict.fromkeys(tokens) if t not in vocab and t not in PUNCT]
    warnings = []
    if unknown:
        warnings.append("Words the model never saw during training: " + ", ".join(unknown) +
                        (". The copy mechanism can still reuse them." if get_model(name)[0].cfg.copy
                         else ". They are read as <UNK>, so the rewrite may lose them."))
    if len(unknown) >= max(3, len(tokens) // 3):
        warnings.append("Many words are unknown: the sentence is far from the training data (English, "
                        "everyday workplace and service situations), so expect a weaker rewrite.")
    if not out:
        warnings.append("The model produced an empty sentence.")
    d = describe(tokens, out)
    if d["repetition"]:
        warnings.append("The output repeats a word: a known weakness of small seq2seq models.")
    if d["politeness_out"] < 0.5 and out:
        warnings.append("The politeness classifier does not find the output polite enough.")
    return {
        "model": name, "decoding": req.decoding, "input_tokens": tokens, "output_tokens": out,
        "output": detokenize(restore_case(out, req.text), capitalize=True),
        "attention": None if r.get("attention") is None else [[round(float(x), 4) for x in row] for row in r["attention"]],
        "p_gen": None if not r.get("p_gen") else [round(p, 3) for p in r["p_gen"]],
        "token_probs": None if probs is None else [round(p, 3) for p in probs],
        "unknown_words": unknown, "warnings": warnings, "latency_ms": round((time.time() - t0) * 1000), **d,
    }


@app.post("/api/compare")
def compare(req: RewriteRequest):
    tokens = validate(req.text)
    rows = [("baseline_copy", "Copy baseline", copy_baseline(tokens)),
            ("baseline_rules", "Rule-based baseline", rules_baseline(tokens))]
    for m in available_models():
        r, _, _ = run_model(m, tokens)
        rows.append((m, MODEL_INFO[m][0], r["tokens"]))
    return {"input_tokens": tokens, "systems": [
        {"id": i, "label": label, "output": detokenize(restore_case(out, req.text), capitalize=True),
         **describe(tokens, out)}
        for i, label, out in rows]}


@app.get("/api/dashboard")
def dashboard():
    results = [r for r in read_json("reports/results.json", []) if "__" not in r["system"] or
               r["system"].endswith(("__beam5", "__sample_t0.7", "__topk5", "__beam3"))]
    curves = {}
    for m in available_models():
        h = read_json(f"experiments/{m}/history.json")
        if h:
            curves[m] = [{k: e[k] for k in ("epoch", "train_loss", "valid_loss", "valid_acc")} for e in h]
    examples = []
    train = ROOT / "data" / "train.csv"
    if train.exists():
        df = pd.read_csv(train)
        for cat, g in df.groupby("category"):
            row = g.sample(1, random_state=3).iloc[0]
            examples.append({"category": cat, "impolite": row["impolite"], "polite": row["polite"]})
    return {"dataset": read_json("reports/dataset_stats.json", {}), "results": results, "curves": curves,
            "examples": examples, "final_system": FINAL_SYSTEM,
            "human": read_json("reports/human_eval_summary.json", {}),
            "vocab": read_json("reports/vocab_stats.json", {}),
            "classifier": read_json("reports/politeness_classifier.json", {}),
            "labels": {m: MODEL_INFO[m][0] for m in MODEL_INFO}}


@app.exception_handler(Exception)
async def unexpected(request, exc):       # never show a Python traceback to the user
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=500, content={"detail": "Something went wrong on the server: "
                                                            f"{type(exc).__name__}. Please try again."})
