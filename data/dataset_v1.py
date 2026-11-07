# =============================================================================
# data/dataset_v1.py
# First version of the dataset, kept to compare it with the final one.
# Regenerates data_v1/, preprocesses it exactly like the final dataset
# and writes reports/dataset_v1_vs_v2.json
#
#   python data/dataset_v1.py
# =============================================================================
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))   # to import preprocessing.py from the project folder

import pandas as pd

from generate_dataset import generate, write
from preprocessing import EOS, Vocab, tokenize, unk_rate

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
    (OUT / "politeness_dataset.csv").unlink()
    train = pd.read_csv(OUT / "train.csv", keep_default_na=False)
    # same preprocessing as for the final dataset
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
