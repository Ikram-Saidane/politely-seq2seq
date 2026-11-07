"""A politeness classifier written from scratch (logistic regression on bag-of-n-grams, numpy only).

It is used as an automatic judge: "does this sentence sound polite?". It is trained on the TRAIN
split only (impolite inputs = class 0, polite references = class 1) and validated on sentences it
never saw, including the held-out phrasings (test_unseen) and the hand-written challenge set, to
check that it does not just recognise the training templates.

    python -m politely.politeness        # trains, evaluates, saves checkpoints/politeness_clf.json
"""
import json
from collections import Counter

import numpy as np

from politely.dataset import ROOT, load_split
from politely.preprocessing import tokenize

PATH = ROOT / "checkpoints" / "politeness_clf.json"


def features(tokens):
    """Unigrams and bigrams (with sentence boundaries), as a set: presence, not counts."""
    toks = ["<s>"] + tokens + ["</s>"]
    return set(tokens) | {a + " " + b for a, b in zip(toks, toks[1:])}


class PolitenessClassifier:
    def __init__(self, vocab=None, w=None, b=0.0):
        self.vocab = vocab or {}
        self.w = np.zeros(len(self.vocab)) if w is None else np.asarray(w)
        self.b = b

    def _matrix(self, sentences):
        X = np.zeros((len(sentences), len(self.vocab)), dtype=np.float32)
        for i, s in enumerate(sentences):
            for f in features(tokenize(s) if isinstance(s, str) else s):
                j = self.vocab.get(f)
                if j is not None:
                    X[i, j] = 1.0
        return X

    def fit(self, sentences, labels, min_count=3, l2=1e-3, lr=0.5, epochs=300):
        counts = Counter(f for s in sentences for f in features(tokenize(s)))
        self.vocab = {f: i for i, f in enumerate(sorted(f for f, c in counts.items() if c >= min_count))}
        X, y = self._matrix(sentences), np.asarray(labels, dtype=np.float32)
        self.w, self.b = np.zeros(X.shape[1], dtype=np.float32), 0.0
        for _ in range(epochs):                    # full-batch gradient descent on the log-loss
            p = 1 / (1 + np.exp(-(X @ self.w + self.b)))
            g = p - y
            self.w -= lr * (X.T @ g / len(y) + l2 * self.w)
            self.b -= lr * g.mean()
        return self

    def predict_proba(self, sentences):
        """P(polite) for each sentence (strings or token lists)."""
        if not len(sentences):
            return np.zeros(0)
        return 1 / (1 + np.exp(-(self._matrix(sentences) @ self.w + self.b)))

    def top_features(self, k=15):
        order = np.argsort(self.w)
        inv = {i: f for f, i in self.vocab.items()}
        return {"polite": [(inv[i], round(float(self.w[i]), 2)) for i in order[::-1][:k]],
                "impolite": [(inv[i], round(float(self.w[i]), 2)) for i in order[:k]]}

    def save(self, path=PATH):
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({"vocab": self.vocab, "w": self.w.tolist(), "b": float(self.b)}), encoding="utf-8")

    @classmethod
    def load(cls, path=PATH):
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(d["vocab"], d["w"], d["b"])


def main():
    train = load_split("train")
    sents = list(train.impolite.drop_duplicates()) + list(train.polite.drop_duplicates())
    labels = [0] * train.impolite.nunique() + [1] * train.polite.nunique()
    clf = PolitenessClassifier().fit(sents, labels)
    clf.save()
    report = {"features": len(clf.vocab), "accuracy": {}}
    for split in ("valid", "test", "test_unseen", "challenge"):
        df = load_split(split)
        imp, pol = list(df.impolite.drop_duplicates()), list(df.polite.drop_duplicates())
        acc_imp = float((clf.predict_proba(imp) < 0.5).mean())
        acc_pol = float((clf.predict_proba(pol) >= 0.5).mean())
        report["accuracy"][split] = {"impolite_recognised": round(acc_imp, 4),
                                     "polite_recognised": round(acc_pol, 4),
                                     "balanced": round((acc_imp + acc_pol) / 2, 4)}
    report["top_features"] = clf.top_features()
    (ROOT / "reports" / "politeness_classifier.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
