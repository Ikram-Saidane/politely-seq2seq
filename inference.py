# =============================================================================
# inference.py
# Generation of a polite sentence with a trained model.
#   greedy : at each step, take the most probable word
#   beam   : keep the k best partial sentences at each step
#   sample : draw the next word at random from the predicted distribution
#            (temperature, top-k or top-p)
#
#   python inference.py "Close the window now."
#   python inference.py "Close the window now." --model B_attention --decoding greedy
# =============================================================================
import argparse
from pathlib import Path

import numpy as np
import torch

from model import Config, build_model
from preprocessing import (DATA_DIR, EOS, ROOT, SOS, Vocab, collate, detokenize, encode_example,
                           restore_case, tokenize)

FINAL_MODEL = "C_copy_nodrop"   # the model chosen on the validation set


def load_model(name, device="cpu"):
    # reloads a saved model: its configuration + its weights
    path = ROOT / "checkpoints" / f"{name}.pt"
    if not path.exists():
        path = Path(name)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = Config.from_dict(ckpt["config"])
    vocab = Vocab.load(DATA_DIR / "vocab.json")
    model = build_model(cfg, len(vocab)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, vocab


def encode_inputs(model, token_lists, vocab, device):
    # sentences -> padded batch -> encoder
    batch = collate([encode_example(t, None, vocab) for t in token_lists])
    src_ext = batch["src_ext"].to(device)
    max_oov = batch["max_oov"] if model.cfg.copy else 0
    enc_out, state, mask, keys = model.encode(batch["src"].to(device), batch["src_len"])
    return batch, src_ext, max_oov, enc_out, state, mask, keys


@torch.no_grad()
def greedy_decode(model, token_lists, vocab, max_len=None, device="cpu", sample=None):
    # Greedy decoding for a list of sentences (as one batch).
    # With sample = {"temperature": .., "top_k": .., "top_p": .., "seed": ..} the next word is
    # drawn at random instead.
    # The decoder starts with <SOS> and stops at <EOS> (or after max_len words).
    max_len = max_len or model.cfg.max_decode_len
    batch, src_ext, max_oov, enc_out, state, mask, keys = encode_inputs(model, token_lists, vocab, device)
    B = len(token_lists)
    gen = torch.Generator().manual_seed(sample.get("seed", 0)) if sample else None
    inp = torch.full((B,), SOS, dtype=torch.long, device=device)
    ids, attns, pgens, probs = [], [], [], []
    done = torch.zeros(B, dtype=torch.bool, device=device)
    for _ in range(max_len):
        log_probs, state, w, pg = model.decoder.step(inp, state, enc_out, keys, mask, src_ext, max_oov)
        if sample:
            logits = log_probs / sample.get("temperature", 1.0)
            if sample.get("top_k"):   # keep only the k most probable words
                kth = logits.topk(sample["top_k"], dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            if sample.get("top_p"):   # keep the most probable words until their total reaches p
                sorted_l, order = logits.sort(dim=-1, descending=True)
                p = sorted_l.softmax(-1)
                remove = p.cumsum(-1) - p > sample["top_p"]
                logits = logits.scatter(1, order, sorted_l.masked_fill(remove, float("-inf")))
            nxt = torch.multinomial(logits.softmax(-1).cpu(), 1, generator=gen).squeeze(1).to(device)
        else:
            nxt = log_probs.argmax(-1)   # greedy: the most probable word
        ids.append(nxt)
        attns.append(w)
        pgens.append(pg)
        probs.append(log_probs.gather(1, nxt.unsqueeze(1)).squeeze(1).exp())
        done |= nxt == EOS
        if done.all():
            break
        inp = model.to_input(nxt)   # the predicted word is the next input

    # cut each sentence after its <EOS>
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
    # Beam search for ONE sentence: at each step, every hypothesis is extended with its k best
    # next words and only the k best hypotheses are kept. The k hypotheses are decoded as a batch.
    max_len = max_len or model.cfg.max_decode_len
    batch, src_ext, max_oov, enc_out, state, mask, keys = encode_inputs(model, [tokens], vocab, device)
    beams = [(0.0, [SOS], [], [])]   # (score, words, attention, p_gen)
    finished = []
    for _ in range(max_len):
        n = len(beams)
        inp = model.to_input(torch.tensor([b[1][-1] for b in beams], device=device))

        def expand(x):
            return None if x is None else x.expand(n, *x.shape[1:])

        log_probs, new_state, w, pg = model.decoder.step(inp, state, expand(enc_out), expand(keys),
                                                         expand(mask), expand(src_ext), max_oov)
        top_lp, top_id = log_probs.topk(k, dim=-1)
        candidates = []
        for i, (score, seq, att, pgs) in enumerate(beams):
            for lp, wid in zip(top_lp[i].tolist(), top_id[i].tolist()):
                candidates.append((score + lp, seq + [wid], i, att, pgs))
        candidates.sort(key=lambda c: c[0], reverse=True)

        beams, keep = [], []
        for score, seq, i, att, pgs in candidates:
            if w is not None:
                att = att + [w[i].cpu().numpy()]
            if pg is not None:
                pgs = pgs + [float(pg[i])]
            if seq[-1] == EOS:
                # divide by the length, otherwise short sentences always win
                finished.append((score / (len(seq) - 1) ** alpha, seq, att, pgs))
            else:
                beams.append((score, seq, att, pgs))
                keep.append(i)
            if len(beams) == k:
                break
        if len(finished) >= k or not beams:
            break
        keep = torch.tensor(keep, device=device)
        if isinstance(new_state, tuple):
            state = tuple(s[:, keep] for s in new_state)
        else:
            state = new_state[:, keep]

    if not finished:
        finished = [(s / (len(q) - 1) ** alpha, q, a, p) for s, q, a, p in beams]
    score, seq, att, pgs = max(finished, key=lambda f: f[0])
    return {"tokens": vocab.decode(seq[1:], oovs=batch["oovs"][0]),
            "attention": np.stack(att) if att else None, "p_gen": pgs or None, "score": score}



def rewrite(text, model, vocab, decoding="beam", beam_size=5, temperature=0.8, seed=0):
    # impolite sentence (a string) -> polite sentence (a string) + details
    tokens = tokenize(text)
    if decoding == "beam":
        result = beam_search(model, tokens, vocab, k=beam_size)
    elif decoding == "sample":
        result = greedy_decode(model, [tokens], vocab, sample={"temperature": temperature, "seed": seed})[0]
    else:
        result = greedy_decode(model, [tokens], vocab)[0]
    sentence = detokenize(restore_case(result["tokens"], text), capitalize=True)
    return sentence, result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("text", nargs="+", help="one or more impolite sentences")
    p.add_argument("--model", default=FINAL_MODEL)
    p.add_argument("--decoding", default="beam", choices=["greedy", "beam", "sample"])
    a = p.parse_args()
    model, vocab = load_model(a.model)
    for text in a.text:
        polite, _ = rewrite(text, model, vocab, a.decoding)
        print(f"{text}\n  -> {polite}")
