"""Shared word vocabulary for the encoder and the decoder.

Special tokens have fixed ids imposed by the course: <PAD>=0, <SOS>=1, <EOS>=2, <UNK>=3.
The vocabulary is built on the TRAIN split only (input and output sides together),
so that validation and test words never leak into the model.
"""
import json
from collections import Counter
from pathlib import Path

PAD, SOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]


class Vocab:
    def __init__(self, itos: list[str] | None = None):
        self.itos: list[str] = list(itos) if itos is not None else list(SPECIALS)
        assert self.itos[:4] == SPECIALS, "special tokens must come first, in the fixed order"
        self.stoi: dict[str, int] = {w: i for i, w in enumerate(self.itos)}

    # ------------------------------------------------------------------ building
    @classmethod
    def build(cls, token_lists, min_freq: int = 2, max_size: int | None = None) -> "Vocab":
        """Keep words seen at least `min_freq` times, most frequent first, at most `max_size` words
        (special tokens not counted). Ties are broken alphabetically so the result is deterministic."""
        counts = Counter(tok for toks in token_lists for tok in toks)
        words = sorted((w for w, c in counts.items() if c >= min_freq and w not in SPECIALS),
                       key=lambda w: (-counts[w], w))
        if max_size is not None:
            words = words[:max_size]
        vocab = cls(SPECIALS + words)
        vocab.counts = counts
        return vocab

    # ------------------------------------------------------------------ mapping
    def __len__(self) -> int:
        return len(self.itos)

    def __contains__(self, word: str) -> bool:
        return word in self.stoi

    def token_to_id(self, word: str) -> int:
        return self.stoi.get(word, UNK)

    def encode(self, tokens: list[str], add_sos: bool = False, add_eos: bool = False) -> list[int]:
        ids = [self.token_to_id(t) for t in tokens]
        if add_sos:
            ids = [SOS] + ids
        if add_eos:
            ids = ids + [EOS]
        return ids

    def decode(self, ids, oovs: list[str] | None = None, strip_special: bool = True) -> list[str]:
        """Map ids back to tokens. Stops at <EOS>; skips <PAD>/<SOS>.

        Ids >= len(vocab) are "extended" ids produced by the copy mechanism: they point
        into `oovs`, the list of source words that are not in the vocabulary."""
        out = []
        for i in ids:
            i = int(i)
            if i == EOS and strip_special:
                break
            if strip_special and i in (PAD, SOS):
                continue
            if i >= len(self.itos):
                out.append(oovs[i - len(self.itos)] if oovs and i - len(self.itos) < len(oovs) else "<UNK>")
            else:
                out.append(self.itos[i])
        return out

    # ------------------------------------------------------------------ io
    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.itos, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Vocab":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


def unk_rate(vocab: Vocab, token_lists) -> float:
    """Share of tokens that would become <UNK>."""
    total = unk = 0
    for toks in token_lists:
        total += len(toks)
        unk += sum(t not in vocab for t in toks)
    return unk / max(total, 1)
