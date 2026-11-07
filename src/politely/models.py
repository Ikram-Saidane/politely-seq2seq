"""Encoder-decoder models, written by hand on top of nn.Embedding / nn.LSTM / nn.GRU / nn.Linear.

The same code builds every model of the report; they differ only by Config flags:

    A  attention=False, copy=False   baseline: the decoder only sees the encoder's final state,
                                     one fixed-size vector that must summarise the whole input
    B  attention=True                + Bahdanau (additive) attention over all encoder states
    C  attention=True,  copy=True    + copy mechanism (pointer-generator, See et al. 2017): the
                                     decoder may copy an input word instead of generating it
    rnn_type="gru"                   the same models with GRU cells instead of LSTM cells

Shapes are batch-first: B = batch, S = source length, T = target length, E = embedding size,
H = hidden size, V = vocabulary size, V' = V + number of source words outside the vocabulary.
Each decoder step returns LOG-probabilities, so the loss is always NLLLoss(ignore_index=PAD);
for A and B this is exactly the token-level cross-entropy.
"""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from politely.vocab import PAD, UNK

EPS = 1e-12
RNN = {"lstm": nn.LSTM, "gru": nn.GRU}


def _top(state):
    """Top-layer hidden state h of an LSTM state (h, c) or a GRU state h: [B, H]."""
    h = state[0] if isinstance(state, tuple) else state
    return h[-1]


class Encoder(nn.Module):
    """Embedding -> (bi)LSTM/GRU over the whole input; padding is skipped thanks to packing."""

    def __init__(self, embedding, cfg):
        super().__init__()
        self.embedding, self.cfg = embedding, cfg
        self.rnn = RNN[cfg.rnn_type](cfg.emb_dim, cfg.hid_dim, cfg.n_layers, batch_first=True,
                                     bidirectional=cfg.bidirectional,
                                     dropout=cfg.dropout if cfg.n_layers > 1 else 0.0)
        self.dropout = nn.Dropout(cfg.dropout)
        self.out_dim = cfg.hid_dim * (2 if cfg.bidirectional else 1)
        if cfg.bidirectional:   # merge the two directions into one state of size H for the decoder
            self.bridge = nn.ModuleList(nn.Linear(2 * cfg.hid_dim, cfg.hid_dim)
                                        for _ in range(2 if cfg.rnn_type == "lstm" else 1))

    def forward(self, src, src_len):
        emb = self.dropout(self.embedding(src))                                   # [B, S, E]
        packed = pack_padded_sequence(emb, src_len.cpu(), batch_first=True, enforce_sorted=False)
        out, state = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=src.size(1))  # [B, S, H*dirs]
        if self.cfg.bidirectional:
            def merge(x, layer):                                                  # [L*2, B, H] -> [L, B, H]
                x = x.view(self.cfg.n_layers, 2, x.size(1), x.size(2))
                return torch.tanh(layer(torch.cat([x[:, 0], x[:, 1]], dim=-1)))
            if self.cfg.rnn_type == "lstm":
                state = (merge(state[0], self.bridge[0]), merge(state[1], self.bridge[1]))
            else:
                state = merge(state, self.bridge[0])
        return out, state


class BahdanauAttention(nn.Module):
    """score(s, h_j) = v^T tanh(W_enc h_j + W_dec s); padding gets -inf before the softmax."""

    def __init__(self, enc_dim, dec_dim, attn_dim):
        super().__init__()
        self.W_enc = nn.Linear(enc_dim, attn_dim, bias=False)
        self.W_dec = nn.Linear(dec_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def project_keys(self, enc_out):
        return self.W_enc(enc_out)                     # computed once per sentence: [B, S, A]

    def forward(self, dec_h, keys, enc_out, mask):
        scores = self.v(torch.tanh(keys + self.W_dec(dec_h).unsqueeze(1))).squeeze(-1)   # [B, S]
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)                                         # sum to 1
        context = torch.bmm(weights.unsqueeze(1), enc_out).squeeze(1)                   # [B, enc_dim]
        return context, weights


class Decoder(nn.Module):
    """Generates the output one token at a time (see `step`)."""

    def __init__(self, embedding, vocab_size, enc_dim, cfg):
        super().__init__()
        self.embedding, self.vocab_size, self.cfg = embedding, vocab_size, cfg
        ctx = enc_dim if cfg.attention else 0
        if cfg.attention:
            self.attention = BahdanauAttention(enc_dim, cfg.hid_dim, cfg.attn_dim)
        # Input at each step: embedding of the previous word (+ context vector with attention).
        self.rnn = RNN[cfg.rnn_type](cfg.emb_dim + ctx, cfg.hid_dim, cfg.n_layers, batch_first=True,
                                     dropout=cfg.dropout if cfg.n_layers > 1 else 0.0)
        # Output layer [s_t ; c_t ; y_{t-1}] -> E -> V (factorised to save parameters).
        self.pre_out = nn.Linear(cfg.hid_dim + ctx + cfg.emb_dim, cfg.emb_dim)
        self.out = nn.Linear(cfg.emb_dim, vocab_size)
        if cfg.copy:
            self.p_gen = nn.Linear(ctx + cfg.hid_dim + cfg.emb_dim, 1)
        self.dropout = nn.Dropout(cfg.dropout)

    def step(self, inp, state, enc_out=None, keys=None, mask=None, src_ext=None, max_oov=0):
        """inp [B] previous token ids (< V). Returns log-probs [B, V'], new state,
        attention weights [B, S] (or None) and p_gen [B, 1] (or None)."""
        emb = self.dropout(self.embedding(inp))                                         # [B, E]
        weights = p_gen = None
        if self.cfg.attention:
            context, weights = self.attention(_top(state), keys, enc_out, mask)
            rnn_in = torch.cat([emb, context], dim=-1)
        else:
            rnn_in = emb
        out, state = self.rnn(rnn_in.unsqueeze(1), state)
        s = out.squeeze(1)                                                              # [B, H]
        feats = [s, emb] + ([context] if self.cfg.attention else [])
        logits = self.out(self.dropout(self.pre_out(torch.cat(feats, dim=-1))))         # [B, V]
        if not self.cfg.copy:
            return F.log_softmax(logits, dim=-1), state, weights, None
        # Pointer-generator: P(w) = p_gen * P_vocab(w) + (1 - p_gen) * attention on copies of w.
        p_vocab = torch.softmax(logits, dim=-1)
        p_gen = torch.sigmoid(self.p_gen(torch.cat([context, s, emb], dim=-1)))         # [B, 1]
        dist = p_gen * p_vocab
        if max_oov > 0:
            dist = torch.cat([dist, dist.new_zeros(dist.size(0), max_oov)], dim=1)       # [B, V']
        dist = dist.scatter_add(1, src_ext, (1 - p_gen) * weights)
        return torch.log(dist + EPS), state, weights, p_gen


class Seq2Seq(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        if cfg.copy and not cfg.attention:
            raise ValueError("the copy mechanism re-uses the attention weights: set attention=True")
        self.cfg, self.vocab_size = cfg, vocab_size
        self.embedding = nn.Embedding(vocab_size, cfg.emb_dim, padding_idx=PAD)   # shared by both sides
        self.encoder = Encoder(self.embedding, cfg)
        self.decoder = Decoder(self.embedding, vocab_size, self.encoder.out_dim, cfg)

    def encode(self, src, src_len):
        enc_out, state = self.encoder(src, src_len)
        mask = src != PAD
        keys = self.decoder.attention.project_keys(enc_out) if self.cfg.attention else None
        return enc_out, state, mask, keys

    def to_input(self, ids):
        """Copied out-of-vocabulary ids cannot be embedded: feed <UNK> to the next step instead."""
        return ids.masked_fill(ids >= self.vocab_size, UNK)

    def forward(self, src, src_len, tgt, tf_ratio=1.0, src_ext=None, max_oov=0):
        """Training pass. tgt [B, T] starts with <SOS>. Returns log-probs [B, T-1, V'] and the
        attention weights [B, T-1, S] (None without attention)."""
        enc_out, state, mask, keys = self.encode(src, src_len)
        inp, outputs, attns = tgt[:, 0], [], []
        for t in range(1, tgt.size(1)):
            log_probs, state, w, _ = self.decoder.step(inp, state, enc_out, keys, mask, src_ext, max_oov)
            outputs.append(log_probs)
            if w is not None:
                attns.append(w)
            # Teacher forcing: with probability tf_ratio feed the TRUE previous word,
            # otherwise the model's own prediction (as at inference time).
            inp = tgt[:, t] if random.random() < tf_ratio else self.to_input(log_probs.argmax(-1))
        return torch.stack(outputs, 1), (torch.stack(attns, 1) if attns else None)


def build_model(cfg, vocab_size) -> Seq2Seq:
    return Seq2Seq(cfg, vocab_size)


def count_parameters(model: Seq2Seq) -> dict:
    def n(params):
        return sum(p.numel() for p in params if p.requires_grad)
    emb = {id(p) for p in model.embedding.parameters()}
    att = {id(p) for p in model.decoder.attention.parameters()} if model.cfg.attention else set()
    counts = {
        "embedding (shared)": n(model.embedding.parameters()),
        "encoder": n(p for p in model.encoder.parameters() if id(p) not in emb),
        "attention": n(p for p in model.decoder.parameters() if id(p) in att),
        "decoder RNN": n(model.decoder.rnn.parameters()),
        "output layer": n(model.decoder.pre_out.parameters()) + n(model.decoder.out.parameters()),
        "copy switch (p_gen)": n(model.decoder.p_gen.parameters()) if model.cfg.copy else 0,
    }
    counts["total"] = n(model.parameters())
    assert counts["total"] == sum(v for k, v in counts.items() if k != "total")
    return counts
