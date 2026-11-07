# Exports the final model (C + copy) and the politeness classifier for the website.
# The website runs the model in the browser (website/js/model.js), so it needs the weights
# as a binary file of float32 values and a small json that says where each matrix is.
#
#   python website/export_model.py              (final model C -> website/model/)
#   python website/export_model.py B_attention  (model B, used for the comparison -> website/model/B/)
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
NAME = sys.argv[1] if len(sys.argv) > 1 else "C_copy_nodrop"
OUT = ROOT / "website" / "model" / ("" if NAME == "C_copy_nodrop" else NAME.split("_")[0])
sys.path.append(str(ROOT))

from evaluate import PUNCT, RUDE_MARKERS, STOP   # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)
ckpt = torch.load(ROOT / "checkpoints" / f"{NAME}.pt", map_location="cpu", weights_only=False)
state = ckpt["state_dict"]

# the embedding is shared by the encoder and the decoder: it is saved once
names = ["embedding.weight",
         "encoder.rnn.weight_ih_l0", "encoder.rnn.weight_hh_l0", "encoder.rnn.bias_ih_l0", "encoder.rnn.bias_hh_l0",
         "decoder.attention.W_enc.weight", "decoder.attention.W_dec.weight", "decoder.attention.W_dec.bias",
         "decoder.attention.v.weight",
         "decoder.rnn.weight_ih_l0", "decoder.rnn.weight_hh_l0", "decoder.rnn.bias_ih_l0", "decoder.rnn.bias_hh_l0",
         "decoder.pre_out.weight", "decoder.pre_out.bias", "decoder.out.weight", "decoder.out.bias",
         "decoder.p_gen.weight", "decoder.p_gen.bias"]
manifest, offset, chunks = {}, 0, []
names = [n for n in names if n in state]   # model B has no copy switch (p_gen)
for n in names:
    a = state[n].detach().numpy().astype("<f4")
    manifest[n] = {"offset": offset, "shape": list(a.shape)}
    offset += a.size
    chunks.append(a.ravel())
np.concatenate(chunks).tofile(OUT / "weights.bin")

vocab = json.loads((ROOT / "data" / "vocab.json").read_text(encoding="utf-8"))
clf = json.loads((ROOT / "checkpoints" / "politeness_clf.json").read_text(encoding="utf-8"))
config = {k: ckpt["config"][k] for k in ("emb_dim", "hid_dim", "attn_dim", "max_decode_len", "copy")}
info = {"config": config, "tensors": manifest, "vocab": vocab,
        "stop": sorted(STOP), "punct": sorted(PUNCT),
        "rude_words": sorted({w for m in RUDE_MARKERS for w in m.split()}),
        "rude_markers": sorted(RUDE_MARKERS, key=len, reverse=True),
        "politeness": {"vocab": clf["vocab"], "w": [round(float(x), 6) for x in clf["w"]], "b": float(clf["b"])}}
(OUT / "model.json").write_text(json.dumps(info, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
print(f"weights.bin: {offset * 4 / 1e6:.1f} MB, model.json: {(OUT / 'model.json').stat().st_size / 1e6:.2f} MB")

# learning curves of A, B and C for the "Training" chart of the website
if NAME == "C_copy_nodrop":
    curves = {}
    for m in ["A_lstm", "B_attention", "C_copy_nodrop"]:
        h = json.loads((ROOT / "experiments" / m / "history.json").read_text(encoding="utf-8"))
        curves[m] = [{"epoch": e["epoch"], "train": round(e["train_loss"], 4), "valid": round(e["valid_loss"], 4)} for e in h]
    (OUT / "curves.json").write_text(json.dumps(curves), encoding="utf-8")
