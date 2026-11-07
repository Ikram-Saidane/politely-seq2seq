"""First version of the dataset (v1), kept so that the improvement can be seen (report, 3.2).

    python -m politely.dataset_v1

1. regenerates the v1 dataset (same seed, v1 building blocks) into data_v1/
2. applies exactly the same preprocessing as for the final dataset (normalisation, lowercasing,
   tokenisation, vocabulary on train only with min_freq 2, numericalisation)
3. writes data_v1/vocab_v1.json, data_v1/train_v1_preprocessed.csv (tokens and ids) and
   reports/dataset_v1_vs_v2.json (the two versions side by side)
"""
import json

import pandas as pd

from politely.dataset import ROOT
from politely.generate_dataset import generate, write
from politely.preprocessing import tokenize
from politely.vocab import EOS, Vocab, unk_rate

OUT = ROOT / "data_v1"


def describe(df, vocab, challenge):
    inputs = df[df.ref_index == 0] if "ref_index" in df else df
    src = [tokenize(x) for x in inputs.impolite]
    words = {t for s in src for t in s}
    return {"inputs": len(inputs), "pairs": len(df), "impolite_word_types": len(words),
            "impolite_len_mean": round(sum(map(len, src)) / len(src), 2),
            "challenge_unk_token_rate": round(unk_rate(vocab, [tokenize(x) for x in challenge.impolite]), 4)}


def main():
    card = write(generate(version=1), out=OUT)
    (OUT / "politeness_dataset.csv").unlink()          # the split files are enough for v1
    train = pd.read_csv(OUT / "train.csv", keep_default_na=False)
    # same preprocessing as build_vocab.py: tokenize, vocabulary on train only, min_freq 2
    sentences = [tokenize(x) for x in train.impolite.drop_duplicates()] + [tokenize(x) for x in train.polite]
    vocab = Vocab.build(sentences, min_freq=2)
    vocab.save(OUT / "vocab_v1.json")
    pre = train.copy()
    pre["impolite_tokens"] = [" ".join(tokenize(x)) for x in pre.impolite]
    pre["impolite_ids"] = [" ".join(map(str, vocab.encode(tokenize(x)) + [EOS])) for x in pre.impolite]
    pre["polite_tokens"] = [" ".join(tokenize(x)) for x in pre.polite]
    pre["polite_ids"] = [" ".join(map(str, vocab.encode(tokenize(x)) + [EOS])) for x in pre.polite]
    pre.to_csv(OUT / "train_v1_preprocessed.csv", index=False, encoding="utf-8")

    challenge = pd.read_csv(ROOT / "data" / "challenge.csv", keep_default_na=False)
    v2_train = pd.read_csv(ROOT / "data" / "train.csv", keep_default_na=False)
    v2_vocab = Vocab.load(ROOT / "data" / "vocab.json")
    comparison = {
        "v1": {"vocab_size": len(vocab), **describe(train, vocab, challenge)},
        "v2": {"vocab_size": len(v2_vocab), **describe(v2_train, v2_vocab, challenge)},
        "v1_pairs_per_split": card["pairs_per_split"],
    }
    (ROOT / "reports" / "dataset_v1_vs_v2.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
