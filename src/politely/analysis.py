"""Dataset statistics and figures (report, section 3).

    python -m politely.analysis
Writes reports/dataset_stats.json and reports/figures/dataset_*.png
"""
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from politely.dataset import ROOT, load_split
from politely.metrics import STOP, PUNCT
from politely.preprocessing import tokenize

FIG = ROOT / "reports" / "figures"
SPLITS = ("train", "valid", "test", "test_unseen", "challenge")
COLORS = {"train": "#2f6f9f", "valid": "#7fa7c9", "test": "#e0a33a", "test_unseen": "#c4553a", "challenge": "#5b8c5a"}


def distinct_ngrams(sentences, n):
    grams = Counter(tuple(s[i:i + n]) for s in sentences for i in range(len(s) - n + 1))
    total = sum(grams.values())
    return len(grams) / total if total else 0.0


def overlap(src, tgt):
    """Share of the polite sentence's tokens that already appear in the impolite one."""
    s = set(src)
    return sum(t in s for t in tgt) / len(tgt)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    data = {s: load_split(s) for s in SPLITS}
    stats = {"splits": {}}
    for s, df in data.items():
        inputs = df.drop_duplicates("impolite")
        src = [tokenize(x) for x in inputs.impolite]
        tgt = [tokenize(x) for x in df.polite]
        pair_src = [tokenize(x) for x in df.impolite]
        stats["splits"][s] = {
            "pairs": len(df), "inputs": len(inputs),
            "references_per_input": round(len(df) / len(inputs), 2),
            "categories": dict(Counter(inputs.category)),
            "impolite_len_mean": round(float(np.mean([len(x) for x in src])), 2),
            "impolite_len_max": int(max(len(x) for x in src)),
            "polite_len_mean": round(float(np.mean([len(x) for x in tgt])), 2),
            "polite_len_max": int(max(len(x) for x in tgt)),
            "token_overlap_mean": round(float(np.mean([overlap(a, b) for a, b in zip(pair_src, tgt)])), 3),
            "distinct_1": round(distinct_ngrams(src, 1), 4), "distinct_2": round(distinct_ngrams(src, 2), 4),
            "distinct_3": round(distinct_ngrams(src, 3), 4),
            "impolite_word_types": len({t for x in src for t in x}),
        }
        if "template_id" in df:
            stats["splits"][s]["impolite_templates"] = int(df.template_id.nunique())
    # duplicates and leakage checks
    all_inputs = {s: set(df.impolite) for s, df in data.items()}
    stats["checks"] = {
        "inputs_shared_between_splits": {f"{a}/{b}": len(all_inputs[a] & all_inputs[b])
                                         for i, a in enumerate(SPLITS) for b in SPLITS[i + 1:]},
        "duplicate_pairs": int(sum(df.duplicated(["impolite", "polite"]).sum() for df in data.values())),
        "pairs_with_identical_input_output": int(sum((df.impolite == df.polite).sum() for df in data.values())),
    }
    train = data["train"]
    content = Counter(t for x in train.impolite.drop_duplicates() for t in tokenize(x) if t not in STOP and t not in PUNCT)
    stats["most_common_content_words_train"] = content.most_common(25)
    (ROOT / "reports" / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    # ---- figure 1: category balance per split
    cats = sorted(train.category.unique())
    fig, ax = plt.subplots(figsize=(9, 3.8))
    width = 0.2
    for i, s in enumerate(("train", "valid", "test", "test_unseen")):
        counts = [stats["splits"][s]["categories"].get(c, 0) for c in cats]
        ax.bar(np.arange(len(cats)) + (i - 1.5) * width, counts, width, label=s, color=COLORS[s])
    ax.set_xticks(range(len(cats)), cats, rotation=25, ha="right")
    ax.set_ylabel("impolite inputs"); ax.set_yscale("log")
    ax.set_title("Inputs per category and split (perfectly balanced)")
    ax.legend(ncols=4, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "dataset_categories.png", dpi=140); plt.close(fig)

    # ---- figure 2: length distributions
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    src = [len(tokenize(x)) for x in train.impolite.drop_duplicates()]
    tgt = [len(tokenize(x)) for x in train.polite]
    bins = np.arange(0, 42, 2)
    ax[0].hist(src, bins=bins, alpha=.75, label="impolite (input)", color="#c4553a")
    ax[0].hist(tgt, bins=bins, alpha=.6, label="polite (target)", color="#2f6f9f")
    ax[0].set_xlabel("tokens"); ax[0].set_ylabel("sentences (train)"); ax[0].legend(); ax[0].set_title("Sentence length")
    ov = [overlap(tokenize(a), tokenize(b)) for a, b in zip(train.impolite, train.polite)]
    ax[1].hist(ov, bins=np.linspace(0, 1, 21), color="#5b8c5a")
    ax[1].set_xlabel("share of target tokens already in the input"); ax[1].set_title("Input / target overlap")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "dataset_lengths_overlap.png", dpi=140); plt.close(fig)
    print(json.dumps({k: v for k, v in stats.items() if k != "most_common_content_words_train"}, indent=1))


if __name__ == "__main__":
    main()
