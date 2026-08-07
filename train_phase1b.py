"""
Phase 1B — corrected training protocol for the Phase 1 ablation.

Phase 1 (the original variant_*.py scripts) trained every variant with a constant
learning rate, no warmup, and reported training loss only. Two consequences:

  1. Transformers are far more warmup-sensitive than GRUs, so a constant LR with no
     warmup systematically handicaps Variant E. The headline "GRU beats Transformer"
     result is therefore confounded with the optimiser schedule.
  2. Training loss is not a valid comparison across variants with different parameter
     counts (A/E ~139.9M vs C/C-corr/D ~152.5M).

This script re-runs the same six architectures with the same seed, data, and step
budget, changing only what was wrong:

  * linear warmup (500 steps) followed by cosine decay to 10% of peak LR
  * a genuine held-out validation split, evaluated every EVAL_EVERY steps
  * train loss, val loss, and LR logged every LOG_EVERY steps

The original scripts are left untouched — they are the record of what Phase 1 actually
ran, and the comparison between the two protocols is itself a result worth reporting.

Usage:
    python train_phase1b.py --variant a
    python train_phase1b.py --variant e --steps 5000

Variants: a, b, c_naive, c_corrected, d, e
"""
import os
import csv
import math
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

# ==========================================
# CONFIG — identical to Phase 1 except the schedule
# ==========================================
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN      = 256
BATCH_SIZE   = 4
TOTAL_STEPS  = 5000
LR           = 3e-4          # now the PEAK lr, not a constant
WARMUP_STEPS = 500           # 10% of budget — the fix for the Transformer baseline
MIN_LR_RATIO = 0.1           # cosine floor, as a fraction of peak
SEED         = 42

LOG_EVERY    = 50
EVAL_EVERY   = 250
EVAL_BATCHES = 50            # 50 x 4 = 200 held-out sequences per evaluation
SAVE_EVERY   = 500

OUT_ROOT     = "checkpoints_phase1b"

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token = tokenizer.eos_token
VOCAB_SIZE = tokenizer.vocab_size


# ==========================================
# VARIANT REGISTRY
# ==========================================
def _load_variant(name):
    """Import a variant module lazily and return (label, model_factory)."""
    if name == "a":
        from variant_a_pure_gru import VariantA
        return "A — Pure GRU", VariantA
    if name == "b":
        from variant_b_per_layer_notepad import VariantB
        return "B — Per-layer notepad", VariantB
    if name == "c_naive":
        from variant_c_naive_shared_notepad import VariantCNaive
        return "C — Shared notepad (naive)", VariantCNaive
    if name == "c_corrected":
        from variant_c_corrected import VariantC
        return "C-corr — Shared notepad (corrected)", VariantC
    if name == "d":
        from variant_d_notepad_attention import VariantD
        return "D — Shared notepad + attention read", VariantD
    if name == "e":
        from variant_e_transformer import VariantE
        return "E — Causal Transformer", VariantE
    raise ValueError(f"unknown variant {name!r}")


VARIANTS = ["a", "b", "c_naive", "c_corrected", "d", "e"]


# ==========================================
# DATA — train + genuine held-out validation
# ==========================================
def collate_fn(batch):
    toks = tokenizer([s["text"] for s in batch],
                     truncation=True, max_length=SEQ_LEN,
                     padding="max_length", return_tensors="pt")
    x = toks.input_ids[:, :-1]
    y = toks.input_ids[:, 1:]
    y[y == tokenizer.pad_token_id] = -100
    return x, y


def build_loaders():
    train_ds = load_dataset("roneneldan/TinyStories", split="train")
    val_ds   = load_dataset("roneneldan/TinyStories", split="validation")

    train_dl = DataLoader(train_ds.shuffle(seed=SEED), batch_size=BATCH_SIZE,
                          collate_fn=collate_fn, num_workers=0)
    # Validation order is fixed (no shuffle) so every eval sees the same sequences
    # and the curve is comparable across steps and across variants.
    val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE,
                          collate_fn=collate_fn, num_workers=0)
    return train_dl, val_dl


# ==========================================
# SCHEDULE — linear warmup then cosine decay
# ==========================================
def build_scheduler(optimizer, total_steps):
    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        progress = min(1.0, progress)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * cosine
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ==========================================
# EVALUATION — the thing Phase 1 never had
# ==========================================
@torch.no_grad()
def evaluate(model, val_dl, loss_fn):
    model.eval()
    total, n = 0.0, 0
    for i, (x, y) in enumerate(val_dl):
        if i >= EVAL_BATCHES:
            break
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        total += loss_fn(logits.view(-1, VOCAB_SIZE), y.view(-1)).item()
        n += 1
    model.train()
    torch.cuda.empty_cache()
    return total / max(1, n)


# ==========================================
# CHECKPOINTS — now including scheduler state
# ==========================================
def save_checkpoint(ckpt_dir, model, optimizer, scheduler, step, train_loss, val_loss):
    path = os.path.join(ckpt_dir, f"ckpt_step{step}.pt")
    torch.save({"step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss}, path)
    print(f"[ckpt] step {step} | train {train_loss:.4f} | val {val_loss:.4f}", flush=True)
    _prune(ckpt_dir, keep=1)


def _prune(ckpt_dir, keep=1):
    files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pt")],
                   key=lambda x: int(x.split("step")[1].split(".")[0]))
    for old in files[:-keep]:
        os.remove(os.path.join(ckpt_dir, old))


def load_latest(ckpt_dir, model, optimizer, scheduler):
    files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pt")],
                   key=lambda x: int(x.split("step")[1].split(".")[0]))
    if not files:
        print("[ckpt] Starting fresh.", flush=True)
        return 0
    ckpt = torch.load(os.path.join(ckpt_dir, files[-1]), map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    print(f"[ckpt] Resumed from step {ckpt['step']}", flush=True)
    return ckpt["step"]


# ==========================================
# TRAINING
# ==========================================
def train(variant, total_steps):
    torch.manual_seed(SEED)
    torch.cuda.empty_cache()

    label, factory = _load_variant(variant)
    ckpt_dir = os.path.join(OUT_ROOT, f"variant_{variant}")
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(ckpt_dir, "loss_log.csv")

    model     = factory().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = build_scheduler(optimizer, total_steps)
    loss_fn   = nn.CrossEntropyLoss(ignore_index=-100)

    p = sum(v.numel() for v in model.parameters())
    print(f"Variant {label} | Params: {p:,} (~{p/1e6:.1f}M)", flush=True)
    print(f"Schedule: warmup {WARMUP_STEPS} -> cosine to {MIN_LR_RATIO:.0%} of {LR}", flush=True)

    start_step = load_latest(ckpt_dir, model, optimizer, scheduler)
    if start_step == 0:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "train_loss", "val_loss", "lr"])

    print("Loading TinyStories (train + validation)...", flush=True)
    train_dl, val_dl = build_loaders()

    print(f"\n--- VARIANT {variant.upper()} (Phase 1B) | {start_step} -> {total_steps} ---",
          flush=True)
    model.train()
    t0        = time.time()
    step      = start_step
    data_iter = iter(train_dl)
    last_val  = float("nan")

    while step < total_steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dl)
            x, y = next(data_iter)

        x, y   = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss   = loss_fn(logits.view(-1, VOCAB_SIZE), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1

        if step % EVAL_EVERY == 0:
            last_val = evaluate(model, val_dl, loss_fn)

        if step % LOG_EVERY == 0:
            lr   = scheduler.get_last_lr()[0]
            vram = torch.cuda.max_memory_allocated(0) / 1024**3 if DEVICE == "cuda" else 0.0
            print(f"Step {step:5d}/{total_steps} | train {loss.item():.4f} | "
                  f"val {last_val:.4f} | lr {lr:.2e} | VRAM {vram:.2f} GB | "
                  f"{time.time()-t0:.0f}s", flush=True)
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([step, round(loss.item(), 4),
                                        round(last_val, 4), f"{lr:.6e}"])
            if DEVICE == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()

        if step % SAVE_EVERY == 0:
            save_checkpoint(ckpt_dir, model, optimizer, scheduler, step,
                            loss.item(), last_val)

    final_val = evaluate(model, val_dl, loss_fn)
    save_checkpoint(ckpt_dir, model, optimizer, scheduler, step, loss.item(), final_val)
    print(f"\nVARIANT {variant.upper()} COMPLETE — final val loss {final_val:.4f}", flush=True)
    return final_val


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase 1B — corrected training protocol")
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--steps", type=int, default=TOTAL_STEPS)
    args = ap.parse_args()
    train(args.variant, args.steps)
