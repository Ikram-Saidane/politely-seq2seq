# =============================================================================
# preprocessing.py
# Step 1 of the pipeline: text -> tokens -> ids -> padded batches
#
#   1. normalisation + lowercasing
#   2. tokenisation (words and punctuation)
#   3. vocabulary built on the training set, with <PAD> <SOS> <EOS> <UNK>
#   4. numerical representation (token ids)
#   5. truncation, padding and batches for PyTorch
#
# Run this file to (re)build the vocabulary:   python preprocessing.py
# =============================================================================
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

# special tokens, always with the same ids
PAD, SOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]


# -----------------------------------------------------------------------------
# 1-2. Normalisation and tokenisation
# -----------------------------------------------------------------------------
# curly quotes and long dashes become simple ones, so the same character always has the same token
QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"})

# a token is a word (with its contraction, like "don't") or one punctuation mark.
# Punctuation is kept as a token because "?" matters a lot: polite requests are often questions.
TOKEN_RE = re.compile(r"\w+(?:'\w+)?|[^\w\s]")


def normalize(text):
    # lowercasing: "Could" and "could" become the same word (less words to learn)
    text = unicodedata.normalize("NFKC", str(text)).translate(QUOTES).lower()
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    # "Send me the report NOW!" -> ['send', 'me', 'the', 'report', 'now', '!']
    return TOKEN_RE.findall(normalize(text))


def detokenize(tokens, capitalize=False):
    # ['could', 'you', 'help', '?'] -> "could you help?"
    text = " ".join(tokens)
    text = re.sub(r" ([.,!?;:%)\]}])", r"\1", text)
    text = re.sub(r"([(\[{$]) ", r"\1", text)
    if capitalize:   # only for display: capital letters at the start of sentences and "I"
        text = re.sub(r"(\w) - (\w)", r"\1-\2", text)
        text = re.sub(r"\bi\b", "I", text)
        text = re.sub(r"\bi'(m|d|ll|ve)\b", lambda m: "I'" + m.group(1), text)
        text = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
    return text


def restore_case(tokens, source_text):
    # for display: give the output words the case they had in the input (tv -> TV, sarah -> Sarah)
    raw = unicodedata.normalize("NFKC", str(source_text)).translate(QUOTES)
    cased = {}
    prev = "."
    for t in TOKEN_RE.findall(raw):
        start_of_sentence = prev in (".", "!", "?", ":")
        special = len(t) > 1 and t[1:].lower() != t[1:]   # TV, iPhone
        if t.lower() != t and (special or not start_of_sentence):
            cased.setdefault(t.lower(), t)
        prev = t
    return [cased.get(t, t) for t in tokens]


# -----------------------------------------------------------------------------
# 3. Vocabulary
# -----------------------------------------------------------------------------
class Vocab:
    # itos = index -> word (a list), stoi = word -> index (a dict)
    def __init__(self, itos=None):
        self.itos = list(itos) if itos is not None else list(SPECIALS)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    @classmethod
    def build(cls, token_lists, min_freq=2):
        # keep the words seen at least min_freq times, the others will be <UNK>
        counts = Counter(tok for toks in token_lists for tok in toks)
        words = sorted((w for w, c in counts.items() if c >= min_freq and w not in SPECIALS),
                       key=lambda w: (-counts[w], w))   # most frequent first
        return cls(SPECIALS + words)

    def __len__(self):
        return len(self.itos)

    def __contains__(self, word):
        return word in self.stoi

    def encode(self, tokens, add_sos=False, add_eos=False):
        # words -> ids, unknown words -> <UNK>
        ids = [self.stoi.get(t, UNK) for t in tokens]
        if add_sos:
            ids = [SOS] + ids
        if add_eos:
            ids = ids + [EOS]
        return ids

    def decode(self, ids, oovs=None):
        # ids -> words, stops at <EOS>.
        # ids >= len(vocab) come from the copy mechanism: they point to a word of the input (oovs)
        out = []
        for i in ids:
            i = int(i)
            if i == EOS:
                break
            if i in (PAD, SOS):
                continue
            if i < len(self.itos):
                out.append(self.itos[i])
            elif oovs and i - len(self.itos) < len(oovs):
                out.append(oovs[i - len(self.itos)])
            else:
                out.append("<UNK>")
        return out

    def save(self, path):
        Path(path).write_text(json.dumps(self.itos, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


def unk_rate(vocab, token_lists):
    # % of tokens that are not in the vocabulary
    total = sum(len(toks) for toks in token_lists)
    unk = sum(t not in vocab for toks in token_lists for t in toks)
    return unk / max(total, 1)


# -----------------------------------------------------------------------------
# Loading the splits
# -----------------------------------------------------------------------------
def load_split(name):
    # name: train, valid, test, test_unseen or challenge
    return pd.read_csv(DATA_DIR / f"{name}.csv", encoding="utf-8", keep_default_na=False)


def load_multiref(name):
    # one entry per impolite input, with all its polite references (used for evaluation)
    df = load_split(name)
    out = []
    for (impolite, category), g in df.groupby(["impolite", "category"], sort=False):
        out.append({"impolite": impolite, "category": category, "references": list(g["polite"])})
    return out


# -----------------------------------------------------------------------------
# 4. Numerical representation
# -----------------------------------------------------------------------------
def extended_ids(src_tokens, tgt_tokens, vocab):
    # For the copy mechanism (model C): a word of the input that is not in the vocabulary
    # (a name, a number like 4521) gets a temporary id len(vocab) + k, so it can still be copied.
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


def encode_example(src_tokens, tgt_tokens, vocab):
    # input: just the ids. target: <SOS> ids <EOS>
    src_ext, tgt_ext, oovs = extended_ids(src_tokens, tgt_tokens, vocab)
    ex = {"src": vocab.encode(src_tokens), "src_ext": src_ext, "oovs": oovs}
    if tgt_tokens is not None:
        ex["tgt"] = vocab.encode(tgt_tokens, add_sos=True, add_eos=True)
        ex["tgt_ext"] = tgt_ext
    return ex


# -----------------------------------------------------------------------------
# 5. Truncation, padding and batches
# -----------------------------------------------------------------------------
class PairDataset(Dataset):
    def __init__(self, df, vocab, max_len=None):
        self.examples = []
        for s, t in zip(df["impolite"], df["polite"]):
            src, tgt = tokenize(s), tokenize(t)
            if max_len:   # truncation (no sentence of our dataset is longer than 40 tokens)
                src, tgt = src[:max_len], tgt[:max_len]
            self.examples.append(encode_example(src, tgt, vocab))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def collate(batch):
    # padding: every sentence of the batch gets the length of the longest one (filled with <PAD>=0)
    def pad(key):
        return pad_sequence([torch.tensor(ex[key]) for ex in batch], batch_first=True, padding_value=PAD)

    out = {"src": pad("src"), "src_ext": pad("src_ext"),
           "src_len": torch.tensor([len(ex["src"]) for ex in batch]),   # real lengths, before padding
           "oovs": [ex["oovs"] for ex in batch],
           "max_oov": max(len(ex["oovs"]) for ex in batch)}
    if "tgt" in batch[0]:
        out["tgt"] = pad("tgt")
        out["tgt_ext"] = pad("tgt_ext")
    return out


def make_loader(df, vocab, batch_size=64, shuffle=False, seed=42, max_len=None):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(PairDataset(df, vocab, max_len), batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate, generator=g)


# -----------------------------------------------------------------------------
# Build the vocabulary (train only) and check the <UNK> rate of each split
# -----------------------------------------------------------------------------
def build_vocab(min_freq=2):
    train = load_split("train")
    # each impolite input appears 2 times in train (2 references), we count it once
    sentences = [tokenize(x) for x in train.impolite.drop_duplicates()] + [tokenize(x) for x in train.polite]
    vocab = Vocab.build(sentences, min_freq=min_freq)
    vocab.save(DATA_DIR / "vocab.json")

    stats = {"min_freq": min_freq, "vocab_size": len(vocab), "unk_rate": {}}
    for split in ("train", "valid", "test", "test_unseen", "challenge"):
        df = load_split(split)
        src = [tokenize(x) for x in df.impolite.drop_duplicates()]
        tgt = [tokenize(x) for x in df.polite]
        stats["unk_rate"][split] = {
            "impolite_tokens": round(unk_rate(vocab, src), 4),
            "polite_tokens": round(unk_rate(vocab, tgt), 4),
            "impolite_sentences_with_unk": round(sum(any(t not in vocab for t in s) for s in src) / len(src), 4)}
    (ROOT / "reports" / "vocab_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return vocab


if __name__ == "__main__":
    vocab = build_vocab()
    example = "Close the window now."
    print("\nExample:", example)
    print("tokens:", tokenize(example))
    print("ids:   ", vocab.encode(tokenize(example), add_eos=True))
