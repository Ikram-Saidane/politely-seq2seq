"""Evaluation metrics (BLEU and content preservation written by hand; chrF from sacrebleu).

Per sentence:
  exact      the output is identical to one of the references (after tokenisation)
  chrf       character n-gram F-score against the best reference (0-1): robust to small variations
  content    share of the input's CONTENT words that are kept in the output. Content words =
             words present in the input AND in every reference, minus function/politeness words:
             the action, object, person, time, number... that a correct paraphrase must keep
  polite     P(polite) given by the from-scratch politeness classifier (politeness.py)
  repetition a word is immediately repeated ("the the")
  success    polite >= 0.5 AND content == 1 AND no repetition: a usable rewrite
Corpus level: BLEU (multi-reference, 1-4 grams, brevity penalty), plus the means of the above.
"""
import math
from collections import Counter

import numpy as np
import sacrebleu

from politely.preprocessing import detokenize

PUNCT = set(".,!?;:'\"()-")
# Function words and politeness markers: not counted as "content".
STOP = set("""a an the to of in on at for from with by and or but so if it its it's is are was were be been
being am i you he she we they me my your our their this that these those do does did have has had will
would could can may might should please thank thanks sorry just again now still yet not no don't won't
can't i'm i'd you're there what when where who why how which as up out about into then than too very
really already""".split())


def ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(hyps, refs, max_n=4):
    """Corpus BLEU in [0, 100] with several references per sentence (clipped n-gram precision,
    geometric mean over n = 1..4, brevity penalty with the closest reference length)."""
    match, total, c, r = [0] * max_n, [0] * max_n, 0, 0
    for hyp, rs in zip(hyps, refs):
        c += len(hyp)
        r += min((abs(len(x) - len(hyp)), len(x)) for x in rs)[1]
        for n in range(1, max_n + 1):
            h = ngrams(hyp, n)
            best = Counter()
            for x in rs:
                best |= ngrams(x, n)
            match[n - 1] += sum(min(v, best[g]) for g, v in h.items())
            total[n - 1] += max(len(hyp) - n + 1, 0)
    if c == 0 or min(match) == 0:
        return 0.0
    log_p = sum(math.log(m / t) for m, t in zip(match, total)) / max_n
    return 100 * (1.0 if c > r else math.exp(1 - r / c)) * math.exp(log_p)


def sacre_bleu(hyps, refs):
    """The same quantity computed by sacrebleu on our tokens (cross-check used in the tests)."""
    k = max(len(r) for r in refs)
    streams = [[" ".join(r[i]) if i < len(r) else None for r in refs] for i in range(k)]
    return sacrebleu.corpus_bleu([" ".join(h) for h in hyps], streams, tokenize="none", force=True).score


def stem(w):
    """Very light stemmer so that 'turn' / 'turning' or 'book' / 'booked' count as the same word
    (polite rewrites often change the verb form: "Close the door" -> "Would you mind closing...")."""
    w = w.lower()
    if w.endswith("'s"):
        w = w[:-2]
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[:-len(suf)]
            break
    if len(w) >= 4 and w[-1] == w[-2] and w[-1] not in "lsz":      # running -> runn -> run
        w = w[:-1]
    return w.rstrip("e") or w                                      # close / closing -> clos


def content_words(src, refs):
    """Content words of the input that ALL references keep (compared after light stemming)."""
    common = {stem(w) for w in src if w not in STOP and w not in PUNCT}
    for r in refs:
        common &= {stem(w) for w in r}
    return common


def has_repetition(tokens):
    return any(a == b and a not in PUNCT for a, b in zip(tokens, tokens[1:]))


def score_sentences(sources, outputs, references, clf):
    """sources/outputs: token lists; references: list of lists of token lists."""
    polite = clf.predict_proba([o for o in outputs]) if outputs else np.zeros(0)
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
