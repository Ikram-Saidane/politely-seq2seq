"""Figures for the report, all drawn from files produced by train.py and evaluate.py.

    python -m politely.figures --best B_attention

Writes into reports/figures/:
    learning_curves.png      validation loss per epoch for every model
    results_success.png      success rate of every system on test / test_unseen / challenge
    metrics_test.png         BLEU, content kept, polite rate for every system on test
    decoding.png             greedy vs beam vs sampling for the best model (validation)
    errors_by_category.png   success rate per category of the best model on each split
    attention_<i>.png        attention maps of the best model on a few test sentences
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from politely.dataset import ROOT
from politely.decode import greedy_decode, load_model
from politely.preprocessing import tokenize

FIG = ROOT / "reports" / "figures"
EXP = ROOT / "experiments"
MODELS = ["A_lstm", "B_attention", "C_copy_nodrop", "C_copy", "D_gru_attention", "E_attention_h128", "E_attention_h512",
          "F_attention_tf0.5"]
COLORS = ["#2f6f9f", "#176f63", "#a87a12", "#d9a441", "#7a4fa3", "#c2476b", "#3f8f4f", "#b4532a"]


def results():
    return json.loads((ROOT / "reports" / "results.json").read_text(encoding="utf-8"))


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)


def learning_curves():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for m, c in zip(MODELS, COLORS):
        f = EXP / m / "history.json"
        if not f.exists():
            continue
        h = json.loads(f.read_text())
        ep = [e["epoch"] for e in h]
        ax[0].plot(ep, [e["valid_loss"] for e in h], color=c, label=m, lw=2)
        ax[1].plot(ep, [e["train_loss"] for e in h], color=c, label=m, lw=2)
    for a, t in zip(ax, ("Validation loss (teacher forcing)", "Training loss")):
        a.set_title(t); a.set_xlabel("epoch"); a.set_ylabel("cross-entropy / token"); clean(a)
    ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "learning_curves.png", dpi=140); plt.close(fig)


def success_bars(rows):
    systems = ["baseline_copy", "baseline_rules"] + [m for m in MODELS if any(r["system"] == m for r in rows)]
    splits = ["test", "test_unseen", "challenge"]
    get = {(r["system"], r["split"]): r for r in rows}
    fig, ax = plt.subplots(figsize=(11, 4.2))
    w = 0.26
    for i, (s, c) in enumerate(zip(splits, ("#176f63", "#e0a33a", "#b4532a"))):
        vals = [get.get((m, s), {}).get("success_rate", np.nan) for m in systems]
        bars = ax.bar(np.arange(len(systems)) + (i - 1) * w, vals, w, label=s, color=c)
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=1)
    ax.set_xticks(range(len(systems)), systems, rotation=20, ha="right")
    ax.set_ylabel("success rate"); ax.set_ylim(0, 1.08)
    ax.set_title("Success = judged polite AND all content words kept AND no repeated word")
    ax.legend(); clean(ax)
    fig.tight_layout(); fig.savefig(FIG / "results_success.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for a, (k, t) in zip(ax, (("BLEU", "BLEU (multi-reference)"), ("content_preservation", "Content kept"),
                              ("polite_rate", "Judged polite"))):
        vals = [get.get((m, "test"), {}).get(k, np.nan) for m in systems]
        a.barh(systems, vals, color=["#999"] * 2 + COLORS[:len(systems) - 2])
        a.invert_yaxis(); a.set_title(t + " — test"); clean(a)
        for y, v in enumerate(vals):
            a.text(v, y, f" {v:.2f}" if k != "BLEU" else f" {v:.1f}", va="center", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "metrics_test.png", dpi=140); plt.close(fig)


def decoding(rows, best):
    decs = ["greedy", "beam3", "beam5", "sample_t0.7", "topk5"]
    get = {r["system"]: r for r in rows if r["split"] == "valid"}
    found = [(d, get.get(best if d == "greedy" else f"{best}__{d}")) for d in decs]
    found = [(d, r) for d, r in found if r]
    if len(found) < 2:
        return
    fig, ax = plt.subplots(figsize=(9, 3.8))
    keys = [("success_rate", "success"), ("content_preservation", "content kept"), ("polite_rate", "polite"),
            ("repetition_rate", "repetitions")]
    w = 0.8 / len(found)
    for i, (d, r) in enumerate(found):
        bars = ax.bar(np.arange(len(keys)) + (i - (len(found) - 1) / 2) * w, [r[k] for k, _ in keys], w, label=d)
        ax.bar_label(bars, fmt="%.2f", fontsize=7)
    ax.set_xticks(range(len(keys)), [n for _, n in keys]); ax.set_ylim(0, 1.1)
    ax.set_title(f"Decoding strategies — {best}, validation set"); ax.legend(fontsize=8, ncols=5); clean(ax)
    fig.tight_layout(); fig.savefig(FIG / "decoding.png", dpi=140); plt.close(fig)


def errors_by_category(best):
    frames = []
    for s in ("test", "test_unseen", "challenge"):
        f = EXP / best / f"{s}_predictions.csv"
        if f.exists():
            df = pd.read_csv(f)
            frames.append(df.groupby("category")["success"].mean().rename(s))
    if not frames:
        return None
    tab = pd.concat(frames, axis=1)
    fig, ax = plt.subplots(figsize=(10, 4))
    tab.plot.bar(ax=ax, color=["#176f63", "#e0a33a", "#b4532a"][:tab.shape[1]], width=0.8)
    ax.set_ylabel("success rate"); ax.set_ylim(0, 1.05); ax.set_xlabel("")
    ax.set_title(f"Success rate per category — {best}")
    ax.tick_params(axis="x", rotation=25); clean(ax)
    fig.tight_layout(); fig.savefig(FIG / "errors_by_category.png", dpi=140); plt.close(fig)
    return tab


def attention_maps(best, n=3):
    model, vocab = load_model(best, "cpu")
    if not model.cfg.attention:
        return
    test = pd.read_csv(ROOT / "data" / "test.csv").drop_duplicates("impolite")
    picks = test.groupby("category").sample(1, random_state=11).head(n)
    for i, text in enumerate(picks.impolite):
        src = tokenize(text)
        r = greedy_decode(model, [src], vocab)[0]
        att = np.array(r["attention"])
        cols = (src + ["<eos>"])[:att.shape[1]]
        rows = (r["tokens"] + ["<eos>"])[:att.shape[0]]
        fig, ax = plt.subplots(figsize=(0.42 * len(cols) + 2, 0.32 * len(rows) + 1.6))
        ax.imshow(att, cmap="Greens", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(cols)), cols, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(rows)), rows, fontsize=8)
        ax.set_xlabel("input (impolite)"); ax.set_ylabel("output (generated)")
        ax.set_title(f"Attention — {best}", fontsize=10)
        fig.tight_layout(); fig.savefig(FIG / f"attention_{i + 1}.png", dpi=140); plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--best", required=True, help="model chosen on the validation set")
    a = p.parse_args(argv)
    FIG.mkdir(parents=True, exist_ok=True)
    rows = results()
    learning_curves(); success_bars(rows); decoding(rows, a.best)
    tab = errors_by_category(a.best)
    if tab is not None:
        tab.round(3).to_csv(ROOT / "reports" / f"success_by_category_{a.best}.csv")
        print(tab.round(3))
    attention_maps(a.best)


if __name__ == "__main__":
    main()
