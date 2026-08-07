# RiverNet — Fixed-Size Gated Memory in Recurrent Language Models

A research project characterizing fixed-size external memory ("notepad") in GRU-based language models.  
All experiments run on a single RTX 3050 Laptop GPU (4 GB VRAM).

---

## What this is

RiverNet attaches a single shared notepad vector to a GRU language model.
The notepad is fixed-size regardless of sequence length — no KV cache, flat VRAM cost.

Phase 1 is a controlled six-variant ablation on TinyStories at 140M parameters.
The goal is one clean, reproducible finding before scaling.

---

## Phase 1 Results

| Variant | Architecture | Final Loss | vs Pure GRU |
|---|---|---|---|
| A | Pure GRU | **2.204** | — |
| B | GRU + per-layer notepad | 2.304 | +0.10 |
| D | GRU + shared notepad + attention read | 2.322 | +0.12 |
| C-corr | GRU + shared notepad (corrected) | 2.523 | +0.32 |
| E | Causal Transformer | 3.266 | +1.06 |
| C | GRU + shared notepad (naive) | 3.460 | +1.26 |

All variants: `d_model=1024`, 6 layers, `batch_size=4`, `seq_len=256`, 5 000 steps, AdamW `lr=3e-4`.

Loss curves:

![Phase 1 loss curves](plots/phase1_loss_curves.png)

---

## Key Findings

1. **GRU beats Transformer at low compute** — ~1.1 nats better at 5 000 steps on TinyStories.
2. **Naive shared notepad is catastrophic** — writing only from the last token position (3.46) is worse than a Transformer.
3. **Write mechanism dominates read mechanism** — attention read (D) still can't recover from a bad write strategy.
4. **Corrected notepad roughly reaches parity** — every-position sequential writes close most of the gap to pure GRU. Reported precisely, because the two ways of measuring it disagree: on **final-step** loss C-corr is **0.32 nats behind** A (2.523 vs 2.204); on the **average of the last 10 logged readings** the gap is **~0.02 nats** (2.35 vs 2.33). Single-step final values are noisy at this logging density, so the averaged figure is the more meaningful of the two — but both are stated here rather than only the flattering one.
5. **Block-level skip connection is architecturally mandatory** — without it, gradient factor $(1-w)^{254} \approx 10^{-77}$ causes complete training failure (loss stuck at ~5.7 for 3 800 steps).

---

## Architecture

### GRUBlock (Variants A, B)
```
GRU(d, d) → LayerNorm(h + x) → MLP(4d) → LayerNorm(mlp + h)
```

### GRUNotepadBlock (Variant C-corrected)
```
all_h = GRU(x)                         # full sequence, one CUDA call
all_r = sigmoid(read_gate(all_h))
all_w = sigmoid(write_gate(all_h))
for t in range(T):
    h_read_t = LN(h_t + r_t * note)
    note     = (1 - w_t) * note + w_t * h_t
out = LN(mlp(h_reads) + h_reads)
out = LN(out + x)                       # block-level skip — REQUIRED
```

---

## Repository structure

```
variant_a_pure_gru.py               # Variant A — pure GRU baseline
variant_b_per_layer_notepad.py      # Variant B — per-layer isolated notepad
variant_c_naive_shared_notepad.py   # Variant C — naive shared notepad (broken design)
variant_c_corrected.py              # Variant C-corr — corrected shared notepad
variant_d_notepad_attention.py      # Variant D — shared notepad + attention read
variant_e_transformer.py            # Variant E — causal Transformer baseline
variant_f_gru_cosine.py             # Variant F — GRU, cosine RL reward (not run — see below)
variant_g_gru_notepad_cosine.py     # Variant G — GRU+notepad, cosine RL reward (not run)
variant_h_gru_logprob.py            # Variant H — GRU, log-prob RL reward (not run)
variant_i_gru_notepad_logprob.py    # Variant I — GRU+notepad, log-prob RL reward (not run)

logs/                               # Summary loss data for each Phase 1 variant
plots/phase1_loss_curves.png        # Figure 1 from the paper
plot_phase1.py                      # Script to regenerate the figure from logs/
findings.md                         # Full results, data quality notes, architecture details
legacy/                             # Earlier exploratory scripts (pre-ablation study)
```

### Note on RL variants (F, G, H, I)

Variants F–I implement teacher-distillation RL training using a frozen gpt-neo-125M teacher.
The scripts are complete and correct but were **not run** — the AdamW optimizer's momentum
buffer initialization requires a contiguous ~198 MB allocation that is unavailable on a 4 GB
Windows GPU without `expandable_segments` support (a Linux-only feature). These variants require
≥8 GB VRAM or a Linux system to execute.

---

## Running the variants

```bash
conda activate hybrid_router
python variant_a_pure_gru.py               # Pure GRU baseline
python variant_b_per_layer_notepad.py      # Per-layer notepad
python variant_c_naive_shared_notepad.py   # Naive shared notepad (reproduces the failure)
python variant_c_corrected.py              # Corrected shared notepad
python variant_d_notepad_attention.py      # Shared notepad + attention read
python variant_e_transformer.py            # Transformer baseline
```

Checkpoints save every 500 steps to `checkpoints/`. Loss logs every 50 steps to `checkpoints/*/loss_log.csv`.  
To regenerate the figure: `python plot_phase1.py`

---

## Hardware

- GPU: NVIDIA RTX 3050 Laptop, 4 GB VRAM
- All Phase 1 variants (A–E) fit within 4 GB with gradient checkpointing enabled
- Estimated training time: ~45–60 min per variant at 5 000 steps

---

## Status

| Phase | Goal | Status |
|---|---|---|
| 1 | Ablation: A vs B vs C vs C-corr vs D vs E | ✅ Complete |
| 2 | Task benchmarks (long-range copy, arithmetic) | Planned |
| 3 | Notepad capacity sweep (d=64/256/512/1024) | Planned |

---

## Known limitations of the Phase 1 protocol

Stated up front, because they bound what these numbers can support:

1. **No LR warmup or schedule.** Every Phase 1 variant used a constant `lr=3e-4`. Transformers
   are substantially more warmup-sensitive than GRUs, so the Variant E baseline is
   disadvantaged by the optimiser configuration, not only by its architecture. **The
   "GRU beats Transformer" result is therefore confounded and should be read as provisional.**
2. **Training loss only.** Phase 1 reported no held-out evaluation, and compares variants with
   differing parameter counts (A/E ~139.9M vs C/C-corr/D ~152.5M).
3. **Small token budget.** `batch_size=4 × seq_len=256 × 5000 steps` ≈ 5M tokens — a fraction
   of one epoch of TinyStories, and a regime that favours recurrent models.
4. **Sparse logging.** The committed `logs/*.csv` contain 5–10 points per variant, not the
   every-50-steps density the training scripts emit.

**Phase 1B** ([`train_phase1b.py`](train_phase1b.py)) re-runs all six architectures with the same
seed, data, and step budget, changing only the protocol: linear warmup (500 steps) into cosine
decay, a genuine held-out validation split evaluated every 250 steps, and full-density logging.
The original Phase 1 scripts are deliberately left untouched as the record of what was run.

```bash
python train_phase1b.py --variant a     # then b, c_naive, c_corrected, d, e
```

Until Phase 1B completes, treat every comparison above as preliminary.

## Paper

Phase 1 is archived on Zenodo: [10.5281/zenodo.20344111](https://doi.org/10.5281/zenodo.20344111) (CC-BY).
Seeking an arXiv endorsement for cs.LG — if you're an eligible endorser and the work looks
sound to you, I'd be grateful for a note.

Full results, data quality notes, and architecture details: [`findings.md`](findings.md)
