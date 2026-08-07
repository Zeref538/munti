"""TinyStories -> tokenizer -> a flat array of token ids on disk.

The whole dataset becomes one long uint16 stream per split, stories separated by
an EOS token. Training then samples a random window out of it. There is no
DataLoader and no padding: every window is exactly block_size tokens of real
text, so no compute is wasted on padding and batching is a single slice.

uint16 holds 0..65535, which covers any vocab we'd use here.

    python -m munti.data prepare              # full dataset (do this on Kaggle)
    python -m munti.data prepare --limit 20000  # a slice, for local dev
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import tokenizer as tk

DATASET = "roneneldan/TinyStories"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DTYPE = np.uint16


def _load(split: str, limit: int | None):
    from datasets import load_dataset

    spec = f"{split}[:{limit}]" if limit else split
    return load_dataset(DATASET, split=spec)


def prepare(limit: int | None = None, vocab_size: int = 4096, tok_sample: int = 50_000):
    """Download TinyStories, train the tokenizer, write train.bin / val.bin."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_ds = _load("train", limit)
    val_ds = _load("validation", limit // 20 if limit else None)
    print(f"stories: train={len(train_ds):,} val={len(val_ds):,}")

    # Train the tokenizer on a sample — merges converge long before the full
    # corpus is consumed, and this keeps the step to seconds instead of minutes.
    sample = train_ds.select(range(min(tok_sample, len(train_ds))))["text"]
    tok = tk.train(sample, vocab_size=vocab_size, out_path=DATA_DIR / "tokenizer.json")
    print(
        f"tokenizer: vocab={tok.get_vocab_size()} "
        f"compression={tk.compression_ratio(tok, sample[:2000]):.2f} chars/token"
    )

    eos_id = tok.token_to_id(tk.EOS)
    for name, ds in (("train", train_ds), ("val", val_ds)):
        ids: list[int] = []
        for enc in tok.encode_batch_fast(ds["text"]):
            ids.extend(enc.ids)
            ids.append(eos_id)
        arr = np.array(ids, dtype=DTYPE)
        arr.tofile(DATA_DIR / f"{name}.bin")
        print(f"{name}.bin: {len(arr):,} tokens")


def load_split(split: str, data_dir: Path | str = DATA_DIR) -> np.ndarray:
    path = Path(data_dir) / f"{split}.bin"
    if not path.exists():
        raise FileNotFoundError(f"No {path}. Run: python -m munti.data prepare")
    # memmap, not read: train.bin can be gigabytes and we only ever touch a few
    # random windows of it per step.
    return np.memmap(path, dtype=DTYPE, mode="r")


def get_batch(data: np.ndarray, batch_size: int, block_size: int, device="cpu", generator=None):
    """Sample `batch_size` random windows. Returns (x, y) where y is x shifted
    one token left — the target for position t is simply the token at t+1."""
    import torch

    high = len(data) - block_size - 1
    assert high > 0, "dataset is smaller than one context window"
    ix = torch.randint(high, (batch_size,), generator=generator)
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack(
        [torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix]
    )
    if str(device).startswith("cuda"):
        # pin + non_blocking: overlap the host->device copy with compute.
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["prepare"])
    p.add_argument("--limit", type=int, default=None, help="stories to use (default: all)")
    p.add_argument("--vocab", type=int, default=4096)
    a = p.parse_args()
    prepare(limit=a.limit, vocab_size=a.vocab)
