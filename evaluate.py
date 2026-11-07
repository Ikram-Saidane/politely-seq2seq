# =============================================================================
# evaluate.py
# Evaluation of the baselines and of the trained models.
#
# Metrics (several, because one metric is not enough for this task):
#   BLEU          n-gram overlap with the polite references (written by hand, checked with sacrebleu)
#   chrF          character n-gram F-score, more tolerant to small changes (sacrebleu)
#   exact match   the output is exactly one of the references
#   content kept  % of the content words of the input that are still in the output
#   politeness    probability "polite" given by our classifier (politeness.py)
#   success       polite AND all content words kept AND no repeated word
#   + token accuracy and perplexity (with teacher forcing)
#
#   python evaluate.py --baselines
#   python evaluate.py --models A_lstm B_attention C_copy_nodrop --decoding greedy
#   python evaluate.py --models C_copy_nodrop --decoding beam5
#   python evaluate.py --table
# =============================================================================
import argparse
import json
import math
import re
import time
from collections import Counter

import numpy as np
import pandas as pd
import sacrebleu
import torch

from inference import beam_search, greedy_decode, load_model
from politeness import PolitenessClassifier
from preprocessing import ROOT, detokenize, load_multiref, load_split, make_loader, tokenize
from train import run_epoch


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
PUNCT = set(".,!?;:'\"()-")
# words that are not "content" (function words, politeness words)
STOP = set("""a an the to of in on at for from with by and or but so if it its it's is are was were be been
being am i you he she we they me my your our their this that these those do does did have has had will
would could can may might should please thank thanks sorry just again now still yet not no don't won't
can't i'm i'd you're there what when where who why how which as up out about into then than too very
really already""".split())


def ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(hyps, refs, max_n=4):
    # BLEU with several references, written by hand (checked against sacrebleu in the tests)
    match, total = [0] * max_n, [0] * max_n
    hyp_len, ref_len = 0, 0
    for hyp, rs in zip(hyps, refs):
        hyp_len += len(hyp)
        ref_len += min((abs(len(x) - len(hyp)), len(x)) for x in rs)[1]   # closest reference length
        for n in range(1, max_n + 1):
            h = ngrams(hyp, n)
            best = Counter()
            for x in rs:
                best |= ngrams(x, n)
            # an n-gram counts at most as many times as it appears in a reference ("clipping")
            match[n - 1] += sum(min(v, best[g]) for g, v in h.items())
            total[n - 1] += max(len(hyp) - n + 1, 0)
    if hyp_len == 0 or min(match) == 0:
        return 0.0
    # geometric mean of the 4 precisions
    log_p = sum(math.log(m / t) for m, t in zip(match, total)) / max_n
    # brevity penalty: too short outputs are penalised
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)
    return 100 * bp * math.exp(log_p)


def sacre_bleu(hyps, refs):
    k = max(len(r) for r in refs)
    streams = [[" ".join(r[i]) if i < len(r) else None for r in refs] for i in range(k)]
    return sacrebleu.corpus_bleu([" ".join(h) for h in hyps], streams, tokenize="none", force=True).score


def stem(w):
    # small stemmer so that "close" and "closing" count as the same word
    w = w.lower()
    if w.endswith("'s"):
        w = w[:-2]
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[:-len(suf)]
            break
    if len(w) >= 4 and w[-1] == w[-2] and w[-1] not in "lsz":   # runn -> run
        w = w[:-1]
    return w.rstrip("e") or w


def content_words(src, refs):
    # words of the input that all the references keep
    common = {stem(w) for w in src if w not in STOP and w not in PUNCT}
    for r in refs:
        common &= {stem(w) for w in r}
    return common


def has_repetition(tokens):
    return any(a == b and a not in PUNCT for a, b in zip(tokens, tokens[1:]))


def score_sentences(sources, outputs, references, clf):
    polite = clf.predict_proba(outputs) if outputs else np.zeros(0)
    rows = []
    for src, out, refs, p in zip(sources, outputs, references, polite):
        keep = content_words(src, refs)
        content = (len(keep & {stem(w) for w in out}) / len(keep)) if keep else 1.0
        rep = has_repetition(out)
        hyp = detokenize(out)
        chrf = sacrebleu.sentence_chrf(hyp, [detokenize(r) for r in refs]).score / 100 if out else 0.0
        rows.append({"exact": int(any(out == r for r in refs)), "chrf": chrf, "content": content,
                     "polite": float(p), "repetition": int(rep), "unk": out.count("<UNK>"), "len": len(out),
                     "success": int(p >= 0.5 and content == 1.0 and not rep and len(out) > 0)})
    return rows


def aggregate(rows, outputs, references):
    n_tok = sum(r["len"] for r in rows)
    mean = lambda k: float(np.mean([r[k] for r in rows]))
    return {
        "BLEU": corpus_bleu(outputs, references),
        "chrF": mean("chrf") * 100,
        "exact_match": mean("exact"),
        "content_preservation": mean("content"),
        "politeness": mean("polite"),
        "polite_rate": float(np.mean([r["polite"] >= 0.5 for r in rows])),
        "repetition_rate": mean("repetition"),
        "unk_rate": sum(r["unk"] for r in rows) / max(n_tok, 1),
        "success_rate": mean("success"),
        "avg_len": n_tok / len(rows),
        "n": len(rows),
    }


# -----------------------------------------------------------------------------
# Baselines (no machine learning)
#   copy  : output = input (keeps all the content but stays impolite)
#   rules : remove the usual rude words and add "please" at the beginning
# -----------------------------------------------------------------------------
RUDE_MARKERS = [
    "and stop wasting my time", "i am not asking twice", "i'm not asking twice", "no excuses",
    "deal with it", "not my problem", "stop asking me about it", "stop asking", "get it right",
    "i don't have all day", "answer me", "hurry up", "end of discussion", "get used to it",
    "this is ridiculous", "unbelievable", "for the last time", "how many times do i have to ask",
    "are you even listening", "it's annoying", "understood", "already", "right now", "immediately",
    "now", "seriously", "honestly", "obviously", "come on", "just", "look", "listen", "well",
    "whatever", "period", "do it", "no way", "forget it", "absolutely not", "typical", "again",
]
_RE = re.compile(r"\b(" + "|".join(re.escape(m) for m in sorted(RUDE_MARKERS, key=len, reverse=True)) + r")\b")


def copy_baseline(tokens):
    return list(tokens)


def rules_baseline(tokens):
    text = _RE.sub(" ", detokenize(tokens))
    out = tokenize(text)
    # remove punctuation left alone after the deletions
    cleaned = []
    for t in out:
        if t in ",.!?" and (not cleaned or cleaned[-1] in ",.!?"):
            continue
        cleaned.append(t)
    return ["please", ","] + cleaned if cleaned else ["please", "."]


# -----------------------------------------------------------------------------
# Running the evaluation
# -----------------------------------------------------------------------------
EXP = ROOT / "experiments"
REPORTS = ROOT / "reports"
EVAL_SPLITS = ("valid", "test", "test_unseen", "challenge")
DECODINGS = {"greedy": {}, "beam3": {"beam": 3}, "beam5": {"beam": 5},
             "sample_t0.7": {"sample": {"temperature": 0.7, "seed": 0}},
             "topk5": {"sample": {"temperature": 1.0, "top_k": 5, "seed": 0}}}


def score_and_save(system, split, data, outputs, extra=None):
    clf = PolitenessClassifier.load()
    sources = [tokenize(ex["impolite"]) for ex in data]
    refs = [[tokenize(r) for r in ex["references"]] for ex in data]
    rows = score_sentences(sources, outputs, refs, clf)
    m = aggregate(rows, outputs, refs)
    m["BLEU_sacrebleu"] = sacre_bleu(outputs, refs)
    m.update(extra or {})
    m.update(system=system, split=split)
    out = EXP / system
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{split}_metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    df = pd.DataFrame(rows)
    df.insert(0, "prediction", [detokenize(o, capitalize=True) for o in outputs])
    df.insert(0, "references", [" ||| ".join(ex["references"]) for ex in data])
    df.insert(0, "impolite", [ex["impolite"] for ex in data])
    df.insert(0, "category", [ex["category"] for ex in data])
    df.to_csv(out / f"{split}_predictions.csv", index=False, encoding="utf-8")
    print(f"{system:28s} {split:12s} BLEU {m['BLEU']:6.2f}  EM {m['exact_match']:.3f}  "
          f"content {m['content_preservation']:.3f}  polite {m['polite_rate']:.3f}  "
          f"success {m['success_rate']:.3f}", flush=True)
    return m


def eval_baselines(splits):
    for split in splits:
        data = load_multiref(split)
        src = [tokenize(ex["impolite"]) for ex in data]
        score_and_save("baseline_copy", split, data, [copy_baseline(s) for s in src])
        score_and_save("baseline_rules", split, data, [rules_baseline(s) for s in src])


@torch.no_grad()
def teacher_forced(model, vocab, split, device):
    loader = make_loader(load_split(split), vocab, 64, max_len=model.cfg.max_len)
    loss, acc = run_epoch(model, loader, device, tf_ratio=1.0)
    return {"perplexity": math.exp(loss), "token_accuracy": acc}


def eval_model(name, decoding, splits, device):
    model, vocab = load_model(name, device)
    opts = DECODINGS[decoding]
    summary = EXP / name / "train_summary.json"
    info = json.loads(summary.read_text(encoding="utf-8")) if summary.exists() else {}
    for split in splits:
        data = load_multiref(split)
        src = [tokenize(ex["impolite"]) for ex in data]
        t0 = time.time()
        if "beam" in opts:
            outs = [beam_search(model, s, vocab, k=opts["beam"], device=device)["tokens"] for s in src]
        else:
            outs = []
            for i in range(0, len(src), 256):
                outs += [r["tokens"] for r in greedy_decode(model, src[i:i + 256], vocab, device=device,
                                                            sample=opts.get("sample"))]
        extra = {"decoding": decoding, "decode_seconds": time.time() - t0, "model": name,
                 "params": info.get("params"), "train_minutes": info.get("train_minutes"),
                 "best_epoch": info.get("best_epoch")}
        if decoding == "greedy" and split != "challenge":
            extra.update(teacher_forced(model, vocab, split, device))
        system = name if decoding == "greedy" else f"{name}__{decoding}"
        score_and_save(system, split, data, outs, extra)


def build_table():
    rows = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(EXP.glob("*/*_metrics.json"))]
    (REPORTS / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lines = ["# Results", "", "_Generated by `python evaluate.py --table`._", ""]
    for split in EVAL_SPLITS:
        sub = sorted((r for r in rows if r["split"] == split),
                     key=lambda r: (not r["system"].startswith("baseline"), r["system"]))
        if not sub:
            continue
        lines += [f"## {split} ({sub[0]['n']} inputs)", "",
                  "| System | BLEU | chrF | Exact match | Content kept | Polite rate | Success | Repetitions | Token acc. | Perplexity |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for r in sub:
            f = lambda k, fmt: fmt.format(r[k]) if r.get(k) is not None else "—"
            lines.append(f"| {r['system']} | {r['BLEU']:.2f} | {r['chrF']:.1f} | {r['exact_match']:.3f} | "
                         f"{r['content_preservation']:.3f} | {r['polite_rate']:.3f} | **{r['success_rate']:.3f}** | "
                         f"{r['repetition_rate']:.3f} | {f('token_accuracy', '{:.3f}')} | {f('perplexity', '{:.2f}')} |")
        lines.append("")
    (REPORTS / "results.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", default=[])
    p.add_argument("--decoding", default="greedy", choices=list(DECODINGS))
    p.add_argument("--splits", nargs="*", default=list(EVAL_SPLITS))
    p.add_argument("--baselines", action="store_true")
    p.add_argument("--table", action="store_true")
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if a.baselines:
        eval_baselines(a.splits)
    for m in a.models:
        eval_model(m, a.decoding, a.splits, device)
    build_table()


if __name__ == "__main__":
    main()
