"""The training loop.

Nothing exotic: sample a batch, predict the next token everywhere at once,
cross-entropy against the true next tokens, AdamW step. The parts that actually
matter for a run finishing successfully are the boring ones —

  warmup      the first few hundred steps use a ramping LR. Adam's variance
              estimates are garbage at step 0; a full-size step then can wreck
              the initialization it never recovers from.
  cosine decay  anneal to ~10% of peak so the model settles instead of
              bouncing around the minimum.
  grad clip   caps the global gradient norm at 1.0. One bad batch producing a
              huge gradient is the usual cause of "loss went to NaN at hour 3".
  checkpoint + resume   Kaggle sessions die at 9 hours. Every checkpoint stores
              the optimizer state and step too, so a resume is exact, not a
              warm restart.

    python -m munti.train --config configs/munti-12m.yaml
    python -m munti.train --config configs/munti-12m.yaml --resume
    python -m munti.train --curve out/loss.csv     # write the plot
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import yaml

from . import data as D
from . import tokenizer as tk
from .model import Munti, MuntiConfig
from .sample import generate_text

# Fixed prompts, sampled at every eval, so the case study can show what the
# model learned over time from an identical starting point.
PROBE_PROMPTS = [
    "Once upon a time, there was a little",
    "Tom and his sister went to the park. They",
    "The dog was very sad because",
]


def lr_at(step: int, *, lr: float, warmup: int, total: int, min_ratio: float = 0.1) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    if step >= total:
        return lr * min_ratio
    progress = (step - warmup) / max(total - warmup, 1)
    return lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def estimate_loss(model, split_data, batch_size, block_size, device, iters=50):
    """Average loss over several batches — a single batch is far too noisy to
    tell whether the model actually improved between evals."""
    model.eval()
    losses = torch.zeros(iters)
    for i in range(iters):
        x, y = D.get_batch(split_data, batch_size, block_size, device)
        _, loss = model(x, y)
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()


def train(config_path: str, resume: bool = False):
    cfg = yaml.safe_load(Path(config_path).read_text())
    out = Path(cfg.get("out_dir", "out"))
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.get("seed", 1337))  # NFR-3: reproducible
    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    bs, block = cfg["batch_size"], cfg["model"]["block_size"]
    total_steps = cfg["max_steps"]

    train_data, val_data = D.load_split("train"), D.load_split("val")
    model = Munti(MuntiConfig(**cfg["model"])).to(device)
    print(f"device={device} params={model.num_params():,}")

    # No weight decay on 1-D params (biases, layernorm gains) — decaying them
    # just shrinks the model's ability to scale activations, for no benefit.
    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.get("weight_decay", 0.1)},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=cfg["lr"], betas=(0.9, 0.95),
    )

    ckpt_path = out / "ckpt.pt"
    start_step = 0
    if resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_step = ck["step"] + 1
        print(f"resumed from step {start_step}")

    log_path = out / "loss.csv"
    if not log_path.exists():
        log_path.write_text("step,train_loss,val_loss,lr\n")
    samples_path = out / "samples.md"

    # Mixed precision. The free Kaggle GPU is a T4 (Turing), which has fp16
    # tensor cores but *no* bf16 — so bf16 there silently costs us the speedup.
    # bf16 where available (no scaler needed, its range matches fp32); fp16 plus
    # a gradient scaler otherwise, because fp16's narrow range underflows small
    # gradients to zero without one.
    # Check the compute capability directly, not is_bf16_supported(): that call
    # counts *emulated* bf16 and returns True on cards with no bf16 hardware at
    # all (it said True on a P100), which would pick a path that crawls. Real
    # bf16 starts at Ampere, sm_80.
    amp_dtype = None
    if device.startswith("cuda"):
        amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype is torch.float16)

    def autocast():
        if amp_dtype is None:
            return nullcontext()
        return torch.amp.autocast("cuda", dtype=amp_dtype)

    print(f"precision: {amp_dtype or 'fp32'}")
    if resume and ckpt_path.exists() and "scaler" in ck:
        scaler.load_state_dict(ck["scaler"])

    t0 = time.time()
    model.train()
    for step in range(start_step, total_steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, lr=cfg["lr"], warmup=cfg["warmup_steps"], total=total_steps)

        x, y = D.get_batch(train_data, bs, block, device)
        with autocast():
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        # Unscale before clipping, or we'd be clipping the scaled gradients and
        # the 1.0 threshold would mean nothing.
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
        scaler.step(opt)
        scaler.update()

        if step % cfg.get("log_every", 100) == 0:
            print(f"step {step:6d} | loss {loss.item():.4f} | {time.time() - t0:.0f}s")

        if (step > 0 and step % cfg["eval_every"] == 0) or step == total_steps - 1:
            with autocast():
                tr = estimate_loss(model, train_data, bs, block, device)
                va = estimate_loss(model, val_data, bs, block, device)
            lr_now = opt.param_groups[0]["lr"]
            print(f"  eval @ {step}: train {tr:.4f} val {va:.4f}")
            with log_path.open("a", newline="") as f:
                csv.writer(f).writerow([step, f"{tr:.4f}", f"{va:.4f}", f"{lr_now:.2e}"])

            torch.save(
                model.checkpoint(
                    opt=opt.state_dict(), scaler=scaler.state_dict(), step=step, val_loss=va
                ),
                ckpt_path,
            )

            # Checkpoint-progression samples (PRD FR-10).
            try:
                tok = tk.load()
                with samples_path.open("a", encoding="utf-8") as f:
                    f.write(f"\n## step {step} (val {va:.4f})\n\n")
                    for prompt in PROBE_PROMPTS:
                        text = generate_text(
                            model, tok, prompt, device=device,
                            max_new_tokens=120, temperature=0.8, top_k=200,
                        )
                        f.write(f"> {text}\n\n")
            except FileNotFoundError:
                pass  # no tokenizer yet (e.g. synthetic-data tests) — not fatal

    print(f"done in {(time.time() - t0) / 60:.1f} min -> {ckpt_path}")


def plot_curve(csv_path="out/loss.csv", png_path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    csv_path = Path(csv_path)
    # Default the image next to its data, so --curve out-nopos/loss.csv doesn't
    # silently overwrite the main run's plot.
    png_path = png_path or csv_path.with_name("curve.png")
    rows = list(csv.DictReader(csv_path.open()))
    steps = [int(r["step"]) for r in rows]
    plt.figure(figsize=(7, 4))
    plt.plot(steps, [float(r["train_loss"]) for r in rows], label="train")
    plt.plot(steps, [float(r["val_loss"]) for r in rows], label="val")
    plt.xlabel("step"); plt.ylabel("cross-entropy loss"); plt.legend()
    plt.title("Munti — training loss"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(png_path, dpi=140)
    print(f"wrote {png_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/munti-12m.yaml")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--curve", nargs="?", const="out/loss.csv", default=None)
    a = p.parse_args()
    if a.curve:
        plot_curve(a.curve)
    else:
        train(a.config, resume=a.resume)
