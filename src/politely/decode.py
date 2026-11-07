"""Inference: greedy decoding (required), beam search and sampling (temperature / top-k / top-p).

greedy   at each step take the single most probable token
beam     keep the k best partial sentences at each step; finished hypotheses are compared with a
         length-normalised score log P(y) / |y|^alpha, otherwise short outputs always win
sample   draw the next token from the (temperature-scaled, top-k / top-p truncated) distribution:
         more varied outputs, at the risk of mistakes
"""
from pathlib import Path

import numpy as np
import torch

from politely.config import Config
from politely.dataset import DATA_DIR, ROOT, collate, encode_example
from politely.models import build_model
from politely.preprocessing import detokenize, tokenize
from politely.vocab import EOS, SOS, Vocab


def load_model(name_or_path, device="cpu"):
    path = ROOT / "checkpoints" / f"{name_or_path}.pt"
    if not path.exists():
        path = Path(name_or_path)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = Config.from_dict(ckpt["config"])
    vocab = Vocab.load(DATA_DIR / "vocab.json")
    model = build_model(cfg, len(vocab)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, vocab


def _encode(model, token_lists, vocab, device):
    batch = collate([encode_example(t, None, vocab) for t in token_lists])
    src_ext = batch["src_ext"].to(device)
    max_oov = batch["max_oov"] if model.cfg.copy else 0
    enc_out, state, mask, keys = model.encode(batch["src"].to(device), batch["src_len"])
    return batch, src_ext, max_oov, enc_out, state, mask, keys


def _select_state(state, idx):
    if isinstance(state, tuple):
        return tuple(s[:, idx] for s in state)
    return state[:, idx]


@torch.no_grad()
def greedy_decode(model, token_lists, vocab, max_len=None, device="cpu", sample=None):
    """Batched greedy decoding, or sampling when `sample` = dict(temperature, top_k, top_p, seed).

    Returns one dict per input: output tokens, attention [len(out)+1, len(in)] (None for model A),
    p_gen per step (model C), and the probability of each generated token."""
    max_len = max_len or model.cfg.max_decode_len
    batch, src_ext, max_oov, enc_out, state, mask, keys = _encode(model, token_lists, vocab, device)
    B = len(token_lists)
    gen = torch.Generator().manual_seed(sample.get("seed", 0)) if sample else None
    inp = torch.full((B,), SOS, dtype=torch.long, device=device)
    ids, attns, pgens, probs = [], [], [], []
    done = torch.zeros(B, dtype=torch.bool, device=device)
    for _ in range(max_len):
        log_probs, state, w, pg = model.decoder.step(inp, state, enc_out, keys, mask, src_ext, max_oov)
        if sample:
            logits = log_probs / max(sample.get("temperature", 1.0), 1e-6)
            if sample.get("top_k"):
                kth = logits.topk(sample["top_k"], dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            if sample.get("top_p"):
                sorted_l, order = logits.sort(dim=-1, descending=True)
                cum = sorted_l.softmax(-1).cumsum(-1)
                remove = cum - sorted_l.softmax(-1) > sample["top_p"]
                logits = logits.scatter(1, order, sorted_l.masked_fill(remove, float("-inf")))
            nxt = torch.multinomial(logits.softmax(-1).cpu(), 1, generator=gen).squeeze(1).to(device)
        else:
            nxt = log_probs.argmax(-1)
        ids.append(nxt); attns.append(w); pgens.append(pg)
        probs.append(log_probs.gather(1, nxt.unsqueeze(1)).squeeze(1).exp())
        done |= nxt == EOS
        if done.all():
            break
        inp = model.to_input(nxt)
    ids = torch.stack(ids, 1).tolist()
    probs = torch.stack(probs, 1).tolist()
    results = []
    for b in range(B):
        seq = ids[b]
        n = seq.index(EOS) + 1 if EOS in seq else len(seq)
        res = {"tokens": vocab.decode(seq[:n], oovs=batch["oovs"][b]), "attention": None, "p_gen": None,
               "token_probs": probs[b][:n]}
        if attns[0] is not None:
            S = int(batch["src_len"][b])
            res["attention"] = torch.stack([a[b, :S] for a in attns[:n]]).cpu().numpy()
        if pgens[0] is not None:
            res["p_gen"] = [float(p[b]) for p in pgens[:n]]
        results.append(res)
    return results


@torch.no_grad()
def beam_search(model, tokens, vocab, k=5, max_len=None, alpha=1.0, device="cpu"):
    """Beam search for ONE sentence (the k hypotheses are processed as a batch of size k)."""
    max_len = max_len or model.cfg.max_decode_len
    batch, src_ext, max_oov, enc_out, state, mask, keys = _encode(model, [tokens], vocab, device)
    beams, finished = [(0.0, [SOS], [], [])], []
    for _ in range(max_len):
        n = len(beams)
        inp = model.to_input(torch.tensor([b[1][-1] for b in beams], device=device))
        exp = lambda x: None if x is None else x.expand(n, *x.shape[1:])
        log_probs, new_state, w, pg = model.decoder.step(inp, state, exp(enc_out), exp(keys), exp(mask),
                                                         exp(src_ext), max_oov)
        top_lp, top_id = log_probs.topk(k, dim=-1)
        cands = []
        for i, (score, seq, att, pgs) in enumerate(beams):
            for lp, wid in zip(top_lp[i].tolist(), top_id[i].tolist()):
                cands.append((score + lp, seq + [wid], i, att, pgs))
        cands.sort(key=lambda c: c[0], reverse=True)
        beams, keep = [], []
        for score, seq, i, att, pgs in cands:
            att = att + ([w[i].cpu().numpy()] if w is not None else [])
            pgs = pgs + ([float(pg[i])] if pg is not None else [])
            if seq[-1] == EOS:
                finished.append((score / (len(seq) - 1) ** alpha, seq, att, pgs))
            else:
                beams.append((score, seq, att, pgs)); keep.append(i)
            if len(beams) == k:
                break
        if len(finished) >= k or not beams:
            break
        state = _select_state(new_state, torch.tensor(keep, device=device))
    if not finished:
        finished = [(s / (len(q) - 1) ** alpha, q, a, p) for s, q, a, p in beams]
    score, seq, att, pgs = max(finished, key=lambda f: f[0])
    return {"tokens": vocab.decode(seq[1:], oovs=batch["oovs"][0]),
            "attention": np.stack(att) if att else None, "p_gen": pgs or None, "score": score}


def generate(model, vocab, texts, strategy="greedy", beam=5, device="cpu", **sample):
    """Raw strings in, (polite strings, raw results) out."""
    toks = [tokenize(t) for t in texts]
    if strategy == "greedy":
        res = greedy_decode(model, toks, vocab, device=device)
    elif strategy == "beam":
        res = [beam_search(model, t, vocab, k=beam, device=device) for t in toks]
    elif strategy == "sample":
        res = greedy_decode(model, toks, vocab, device=device, sample=sample or {"temperature": 0.8})
    else:
        raise ValueError(strategy)
    return [detokenize(r["tokens"], capitalize=True) for r in res], res
