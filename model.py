# =============================================================================
# model.py
# The encoder-decoder, written with PyTorch (nn.Embedding, nn.LSTM, nn.Linear).
#
#   Model A : embedding -> LSTM encoder -> LSTM decoder -> linear -> softmax
#   Model B : A + Bahdanau attention
#   Model C : B + copy mechanism (pointer-generator)
#
# All the models use the same classes, only the configuration changes
# (one json file per model in configs/).
# =============================================================================
import json
import random
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from preprocessing import PAD, UNK


# -----------------------------------------------------------------------------
# Hyper-parameters
# -----------------------------------------------------------------------------
@dataclass
class Config:
    name: str = "A_lstm"
    # model
    rnn_type: str = "lstm"     # "lstm" or "gru"
    emb_dim: int = 128         # size of the word vectors
    hid_dim: int = 256         # size of the LSTM hidden state
    dropout: float = 0.3
    attention: bool = False    # model B
    copy: bool = False         # model C (needs attention)
    attn_dim: int = 128
    # data
    min_freq: int = 2
    max_len: int = 40          # truncation
    max_decode_len: int = 40   # the decoder stops after 40 words if it never produces <EOS>
    # training
    batch_size: int = 64
    lr: float = 1e-3
    clip: float = 1.0          # gradient clipping
    tf_ratio: float = 1.0      # teacher forcing ratio
    epochs: int = 20
    patience: int = 3          # early stopping
    lr_patience: int = 1
    lr_factor: float = 0.5
    seed: int = 42
    unk_dropout: float = 0.0   # % of input words replaced by <UNK> during training (model C')

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


RNN = {"lstm": nn.LSTM, "gru": nn.GRU}


def last_hidden(state):
    # an LSTM state is (h, c), a GRU state is only h
    h = state[0] if isinstance(state, tuple) else state
    return h[-1]


# -----------------------------------------------------------------------------
# Encoder: reads the impolite sentence
# -----------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, embedding, cfg):
        super().__init__()
        self.embedding = embedding
        # one layer, unidirectional
        self.rnn = RNN[cfg.rnn_type](cfg.emb_dim, cfg.hid_dim, batch_first=True)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, src, src_len):
        emb = self.dropout(self.embedding(src))            # [batch, src_len, emb_dim]
        # "packing" tells the LSTM the real length of each sentence, so it skips the padding
        packed = pack_padded_sequence(emb, src_len.cpu(), batch_first=True, enforce_sorted=False)
        out, state = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=src.size(1))
        # out   = one hidden state per input word (used by the attention)
        # state = last hidden state = representation of the whole sentence (initial decoder state)
        return out, state


# -----------------------------------------------------------------------------
# Bahdanau attention
# -----------------------------------------------------------------------------
class BahdanauAttention(nn.Module):
    # With one fixed vector, the decoder forgets parts of the input.
    # With attention, at each step the decoder looks at ALL the input words:
    #   score_j  = v^T tanh(W_enc h_j + W_dec s)     (h_j = encoder state of word j, s = decoder state)
    #   weights  = softmax(scores)                   (they sum to 1)
    #   context  = sum_j weights_j * h_j
    def __init__(self, enc_dim, dec_dim, attn_dim):
        super().__init__()
        self.W_enc = nn.Linear(enc_dim, attn_dim, bias=False)
        self.W_dec = nn.Linear(dec_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def project_keys(self, enc_out):
        # W_enc h_j does not change during decoding, so it is computed only once per sentence
        return self.W_enc(enc_out)

    def forward(self, dec_h, keys, enc_out, mask):
        scores = self.v(torch.tanh(keys + self.W_dec(dec_h).unsqueeze(1))).squeeze(-1)   # [batch, src_len]
        scores = scores.masked_fill(~mask, float("-inf"))   # no attention on <PAD>
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), enc_out).squeeze(1)                    # [batch, hid_dim]
        return context, weights


# -----------------------------------------------------------------------------
# Decoder: writes the polite sentence one word at a time
# -----------------------------------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, embedding, vocab_size, cfg):
        super().__init__()
        self.embedding = embedding
        self.cfg = cfg
        ctx = cfg.hid_dim if cfg.attention else 0
        if cfg.attention:
            self.attention = BahdanauAttention(cfg.hid_dim, cfg.hid_dim, cfg.attn_dim)
        # input at each step = embedding of the previous word (+ context vector with attention)
        self.rnn = RNN[cfg.rnn_type](cfg.emb_dim + ctx, cfg.hid_dim, batch_first=True)
        # output layer: [decoder state ; previous word ; context] -> emb_dim -> vocabulary
        self.pre_out = nn.Linear(cfg.hid_dim + ctx + cfg.emb_dim, cfg.emb_dim)
        self.out = nn.Linear(cfg.emb_dim, vocab_size)
        if cfg.copy:
            self.p_gen = nn.Linear(ctx + cfg.hid_dim + cfg.emb_dim, 1)
        self.dropout = nn.Dropout(cfg.dropout)

    def step(self, inp, state, enc_out=None, keys=None, mask=None, src_ext=None, max_oov=0):
        # ONE decoding step: previous word -> log-probabilities of the next word
        emb = self.dropout(self.embedding(inp))
        weights = p_gen = None
        if self.cfg.attention:
            context, weights = self.attention(last_hidden(state), keys, enc_out, mask)
            rnn_in = torch.cat([emb, context], dim=-1)
        else:
            rnn_in = emb
        out, state = self.rnn(rnn_in.unsqueeze(1), state)
        s = out.squeeze(1)
        feats = [s, emb] + ([context] if self.cfg.attention else [])
        logits = self.out(self.dropout(self.pre_out(torch.cat(feats, dim=-1))))   # one score per word
        if not self.cfg.copy:
            return F.log_softmax(logits, dim=-1), state, weights, None

        # Copy mechanism: p_gen (between 0 and 1) decides between
        #   generating a word of the vocabulary   (probability p_gen * P_vocab)
        #   copying a word of the input           (probability (1 - p_gen) * attention weight)
        p_vocab = torch.softmax(logits, dim=-1)
        p_gen = torch.sigmoid(self.p_gen(torch.cat([context, s, emb], dim=-1)))
        dist = p_gen * p_vocab
        if max_oov > 0:   # extra places for the input words that are not in the vocabulary
            dist = torch.cat([dist, dist.new_zeros(dist.size(0), max_oov)], dim=1)
        dist = dist.scatter_add(1, src_ext, (1 - p_gen) * weights)
        return torch.log(dist + 1e-12), state, weights, p_gen


# -----------------------------------------------------------------------------
# Full model
# -----------------------------------------------------------------------------
class Seq2Seq(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        # one embedding shared by the encoder and the decoder (same language on both sides)
        self.embedding = nn.Embedding(vocab_size, cfg.emb_dim, padding_idx=PAD)
        self.encoder = Encoder(self.embedding, cfg)
        self.decoder = Decoder(self.embedding, vocab_size, cfg)

    def encode(self, src, src_len):
        enc_out, state = self.encoder(src, src_len)
        mask = src != PAD
        keys = self.decoder.attention.project_keys(enc_out) if self.cfg.attention else None
        return enc_out, state, mask, keys

    def to_input(self, ids):
        # a copied word that is not in the vocabulary can't be embedded: the next step gets <UNK>
        return ids.masked_fill(ids >= self.vocab_size, UNK)

    def forward(self, src, src_len, tgt, tf_ratio=1.0, src_ext=None, max_oov=0):
        # used for training: tgt = <SOS> w1 w2 ... <EOS>
        enc_out, state, mask, keys = self.encode(src, src_len)
        inp = tgt[:, 0]   # first input of the decoder = <SOS>
        outputs, attns = [], []
        for t in range(1, tgt.size(1)):
            log_probs, state, w, _ = self.decoder.step(inp, state, enc_out, keys, mask, src_ext, max_oov)
            outputs.append(log_probs)
            if w is not None:
                attns.append(w)
            # teacher forcing: with probability tf_ratio we give the TRUE previous word,
            # otherwise the word predicted by the model
            if random.random() < tf_ratio:
                inp = tgt[:, t]
            else:
                inp = self.to_input(log_probs.argmax(-1))
        attns = torch.stack(attns, 1) if attns else None
        return torch.stack(outputs, 1), attns


def build_model(cfg, vocab_size):
    if cfg.copy and not cfg.attention:
        raise ValueError("the copy mechanism needs attention")
    return Seq2Seq(cfg, vocab_size)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
