"""Build the shared vocabulary on the TRAIN split only and report <UNK> rates per split.

    python -m politely.build_vocab
"""
import json

from politely.dataset import DATA_DIR, ROOT, load_split
from politely.preprocessing import tokenize
from politely.vocab import Vocab, unk_rate


def main(min_freq=2):
    train = load_split("train")
    # each impolite input appears once per reference: count it once
    sentences = [tokenize(x) for x in train.impolite.drop_duplicates()] + [tokenize(x) for x in train.polite]
    vocab = Vocab.build(sentences, min_freq=min_freq)
    vocab.save(DATA_DIR / "vocab.json")
    stats = {"min_freq": min_freq, "vocab_size": len(vocab), "unk_rate": {}}
    for split in ("train", "valid", "test", "test_unseen", "challenge"):
        df = load_split(split)
        src = [tokenize(x) for x in df.impolite.drop_duplicates()]
        tgt = [tokenize(x) for x in df.polite]
        stats["unk_rate"][split] = {
            "impolite_tokens": round(unk_rate(vocab, src), 4), "polite_tokens": round(unk_rate(vocab, tgt), 4),
            "impolite_sentences_with_unk": round(sum(any(t not in vocab for t in s) for s in src) / len(src), 4)}
    (ROOT / "reports" / "vocab_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return vocab


if __name__ == "__main__":
    main()
