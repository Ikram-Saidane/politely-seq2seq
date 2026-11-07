"""Loading the splits, encoding pairs as tensors, and batching.

Each example carries two encodings:
* the plain encoding, where unknown words become <UNK> (decoder inputs, models without copy);
* the "extended" encoding used by the copy mechanism: a source word that is not in the vocabulary
  (a reference number such as "4521", a rare name) gets a temporary id len(vocab) + k, so the
  model can still point to it and copy it into the output.
"""
from pathlib import Path

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from politely.preprocessing import tokenize
from politely.vocab import EOS, PAD, SOS, UNK, Vocab

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
SPLITS = ("train", "valid", "test", "test_unseen", "challenge")


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"{name}.csv", encoding="utf-8", keep_default_na=False)


def load_multiref(name: str) -> list[dict]:
    """One entry per distinct impolite input, with all its polite references (for evaluation)."""
    df = load_split(name)
    out = []
    for (impolite, category), g in df.groupby(["impolite", "category"], sort=False):
        out.append({"impolite": impolite, "category": category, "references": list(g["polite"]),
                    "template_id": g["template_id"].iloc[0] if "template_id" in g else "handwritten"})
    return out


def extended_ids(src_tokens, tgt_tokens, vocab: Vocab):
    oovs, src_ext = [], []
    for t in src_tokens:
        if t in vocab:
            src_ext.append(vocab.stoi[t])
        else:
            if t not in oovs:
                oovs.append(t)
            src_ext.append(len(vocab) + oovs.index(t))
    tgt_ext = None
    if tgt_tokens is not None:
        tgt_ext = [SOS]
        for t in tgt_tokens:
            if t in vocab:
                tgt_ext.append(vocab.stoi[t])
            elif t in oovs:
                tgt_ext.append(len(vocab) + oovs.index(t))
            else:
                tgt_ext.append(UNK)
        tgt_ext.append(EOS)
    return src_ext, tgt_ext, oovs


def encode_example(src_tokens, tgt_tokens, vocab: Vocab) -> dict:
    src_ext, tgt_ext, oovs = extended_ids(src_tokens, tgt_tokens, vocab)
    ex = {"src": vocab.encode(src_tokens), "src_ext": src_ext, "oovs": oovs, "src_tokens": src_tokens}
    if tgt_tokens is not None:
        ex["tgt"] = vocab.encode(tgt_tokens, add_sos=True, add_eos=True)
        ex["tgt_ext"] = tgt_ext
    return ex


class PairDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocab: Vocab, max_len: int | None = None):
        self.examples = []
        for s, t in zip(df["impolite"], df["polite"]):
            src, tgt = tokenize(s), tokenize(t)
            if max_len:                                   # truncation (see report, section 4)
                src, tgt = src[:max_len], tgt[:max_len]
            self.examples.append(encode_example(src, tgt, vocab))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def collate(batch: list[dict]) -> dict:
    """Pad every sequence to the longest one of the batch (batch_first) and keep real lengths."""
    def pad(key):
        return pad_sequence([torch.tensor(ex[key], dtype=torch.long) for ex in batch],
                            batch_first=True, padding_value=PAD)

    out = {"src": pad("src"), "src_ext": pad("src_ext"),
           "src_len": torch.tensor([len(ex["src"]) for ex in batch]),
           "oovs": [ex["oovs"] for ex in batch], "max_oov": max(len(ex["oovs"]) for ex in batch)}
    if "tgt" in batch[0]:
        out["tgt"], out["tgt_ext"] = pad("tgt"), pad("tgt_ext")
    return out


def make_loader(df, vocab, batch_size=64, shuffle=False, seed=42, max_len=None) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(PairDataset(df, vocab, max_len), batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate, generator=g)
