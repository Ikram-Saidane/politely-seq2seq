"""All hyper-parameters of an experiment, in one place (one JSON file per experiment in configs/)."""
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class Config:
    name: str = "A_lstm"
    # ---- model
    rnn_type: str = "lstm"        # "lstm" or "gru"
    emb_dim: int = 128
    hid_dim: int = 256
    n_layers: int = 1
    dropout: float = 0.3
    bidirectional: bool = False
    attention: bool = False       # model B
    copy: bool = False            # model C (requires attention)
    attn_dim: int = 128
    # ---- data
    min_freq: int = 2             # words seen fewer times in train become <UNK>
    max_len: int = 40             # truncation length in tokens (no sentence of the dataset is longer)
    max_decode_len: int = 40
    # ---- optimisation
    batch_size: int = 64
    lr: float = 1e-3
    clip: float = 1.0
    tf_ratio: float = 1.0         # teacher forcing probability
    epochs: int = 20
    patience: int = 3             # early stopping on the validation loss
    lr_patience: int = 1
    lr_factor: float = 0.5
    seed: int = 42
    # Word dropout: share of input words replaced by <UNK> during training only. With the copy
    # mechanism the model then learns to copy words it cannot read (unknown words, names, numbers).
    unk_dropout: float = 0.0

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, d):
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
