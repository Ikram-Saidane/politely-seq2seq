# =============================================================================
# train.py
# Training of one model.
#   loss      : token-level cross-entropy (NLLLoss on log-probabilities, <PAD> ignored)
#   optimizer : Adam, learning rate halved when the validation loss stops improving
#   early stopping on the validation loss, the best model is saved in checkpoints/<name>.pt
#
#   python train.py --config configs/C_copy_nodrop.json
#   python train.py --config configs/C_copy_nodrop.json --overfit 40     (sanity check)
# =============================================================================
import argparse
import json
import math
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from model import Config, build_model, count_parameters
from preprocessing import DATA_DIR, PAD, ROOT, UNK, Vocab, load_split, make_loader

EXP_DIR = ROOT / "experiments"
CKPT_DIR = ROOT / "checkpoints"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def batch_forward(model, batch, device, tf_ratio):
    # runs the model on one batch and returns (predictions, gold words)
    src = batch["src"].to(device)
    tgt = batch["tgt"].to(device)
    p = model.cfg.unk_dropout
    if model.training and p > 0:
        # word dropout (only model C'): hide some input words behind <UNK>
        drop = (torch.rand(src.shape, device=device) < p) & (src > UNK)
        src = src.masked_fill(drop, UNK)
    max_oov = batch["max_oov"] if model.cfg.copy else 0
    log_probs, _ = model(src, batch["src_len"], tgt, tf_ratio, batch["src_ext"].to(device), max_oov)
    # the model must predict the target without its first token <SOS>
    if model.cfg.copy:
        gold = batch["tgt_ext"].to(device)[:, 1:]
    else:
        gold = tgt[:, 1:]
    return log_probs, gold


def run_epoch(model, loader, device, tf_ratio, optimizer=None, clip=1.0):
    # one pass over the data (training if an optimizer is given, otherwise evaluation)
    # returns the mean loss per token and the token accuracy
    train = optimizer is not None
    model.train(train)
    criterion = nn.NLLLoss(ignore_index=PAD, reduction="sum")
    total_loss, n_tok, n_correct = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            log_probs, gold = batch_forward(model, batch, device, tf_ratio)
            loss = criterion(log_probs.reshape(-1, log_probs.size(-1)), gold.reshape(-1))
            mask = gold != PAD
            tokens = int(mask.sum())
            if train:
                optimizer.zero_grad()
                (loss / tokens).backward()   # backpropagation
                nn.utils.clip_grad_norm_(model.parameters(), clip)   # avoids exploding gradients
                optimizer.step()
            total_loss += loss.item()
            n_tok += tokens
            n_correct += int(((log_probs.argmax(-1) == gold) & mask).sum())
    return total_loss / n_tok, n_correct / n_tok


def plot_curves(history, path, title):
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(ep, [h["train_loss"] for h in history], marker="o", label="train")
    ax[0].plot(ep, [h["valid_loss"] for h in history], marker="o", label="validation")
    ax[0].set_ylabel("loss")
    ax[1].plot(ep, [h["train_acc"] for h in history], marker="o", label="train")
    ax[1].plot(ep, [h["valid_acc"] for h in history], marker="o", label="validation")
    ax[1].set_ylabel("token accuracy")
    for a in ax:
        a.set_xlabel("epoch")
        a.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def train(cfg, overfit=0):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocab.load(DATA_DIR / "vocab.json")
    name = f"sanity_{cfg.name}" if overfit else cfg.name
    out = EXP_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(exist_ok=True)
    cfg.save(out / "config.json")
    log_file = open(out / "train_log.txt", "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")

    train_df = load_split("train")
    valid_df = load_split("valid")
    if overfit:   # the model should be able to memorise a few pairs
        train_df = valid_df = train_df.iloc[:overfit]
    train_loader = make_loader(train_df, vocab, cfg.batch_size, shuffle=True, seed=cfg.seed, max_len=cfg.max_len)
    valid_loader = make_loader(valid_df, vocab, cfg.batch_size, max_len=cfg.max_len)

    model = build_model(cfg, len(vocab)).to(device)
    n_params = count_parameters(model)
    log(f"{name}: device={device} parameters={n_params:,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=cfg.lr_factor,
                                                           patience=cfg.lr_patience)
    history = []
    best = math.inf
    bad_epochs = 0
    t_start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, device, cfg.tf_ratio, optimizer, cfg.clip)
        # validation with teacher forcing = 1 so that it is comparable between epochs
        va_loss, va_acc = run_epoch(model, valid_loader, device, tf_ratio=1.0)
        if not overfit:
            scheduler.step(va_loss)
        lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": tr_loss, "valid_loss": va_loss, "train_acc": tr_acc,
                        "valid_acc": va_acc, "train_ppl": math.exp(tr_loss), "valid_ppl": math.exp(va_loss),
                        "lr": lr, "seconds": time.time() - t0})
        improved = va_loss < best - 1e-4
        if improved:   # keep the best model on validation
            best = va_loss
            bad_epochs = 0
            torch.save({"config": cfg.to_dict(), "state_dict": model.state_dict(), "epoch": epoch,
                        "valid_loss": va_loss}, CKPT_DIR / f"{name}.pt")
        else:
            bad_epochs += 1
        log(f"epoch {epoch:2d}  train {tr_loss:.4f} acc {tr_acc:.3f}  valid {va_loss:.4f} acc {va_acc:.3f} "
            f"ppl {math.exp(va_loss):6.2f}  lr {lr:.1e}  {time.time() - t0:.0f}s{'  *' if improved else ''}")
        (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        plot_curves(history, out / "curves.png", name)
        if not overfit and bad_epochs >= cfg.patience:
            log("early stopping")
            break

    best_h = min(history, key=lambda h: h["valid_loss"])
    summary = {"name": name, "best_epoch": best_h["epoch"], "epochs_run": len(history),
               "best_valid_loss": best_h["valid_loss"], "best_valid_ppl": best_h["valid_ppl"],
               "best_valid_token_acc": best_h["valid_acc"], "train_minutes": (time.time() - t_start) / 60,
               "device": str(device), "params": n_params}
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(json.dumps(summary))
    log_file.close()
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--overfit", type=int, default=0)
    p.add_argument("--epochs", type=int)
    a = p.parse_args()
    cfg = Config.load(a.config)
    if a.overfit:   # memorisation test: no dropout, full teacher forcing
        cfg.dropout, cfg.tf_ratio, cfg.batch_size, cfg.epochs = 0.0, 1.0, 20, a.epochs or 60
    elif a.epochs:
        cfg.epochs = a.epochs
    train(cfg, overfit=a.overfit)
