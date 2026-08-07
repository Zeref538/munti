# Munti

> *Munti* — Filipino for **tiny**.

A ~12.5M-parameter language model written from scratch in pure PyTorch and
trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
on free hardware. The tokenizer, the transformer, the training loop and the
sampler are all implemented here — no pretrained weights, no ready-made GPT
class.

**This is a learning artifact, not a product.** It cannot answer questions, hold
a conversation, or recall facts. It writes short children's-story prose. The
point is demonstrated understanding of how a language model works, plus an
honest account of what a model this small can and cannot do.

**Status:** Phase 1 complete — model, training loop and sampling built, and the
correctness gate passes. Full training run pending.

## What's here

| path | what |
|---|---|
| [munti/tokenizer.py](munti/tokenizer.py) | byte-level BPE, trained on our own corpus |
| [munti/data.py](munti/data.py) | TinyStories → flat `uint16` token stream on disk |
| [munti/model.py](munti/model.py) | the transformer, written and annotated by hand |
| [munti/train.py](munti/train.py) | training loop, warmup+cosine, checkpoint/resume |
| [munti/sample.py](munti/sample.py) | temperature + top-k generation |
| [test_munti.py](test_munti.py) | the correctness gate — run this first |
| [configs/](configs/) | one YAML per run, including the ablation |

## Architecture

Decoder-only transformer, pre-LN, weight-tied embeddings.

| | |
|---|---|
| params | ~12.5M |
| layers / heads / width | 6 / 6 / 384 |
| context | 256 tokens |
| vocab | 4096 (byte-level BPE) |
| compression | 3.98 chars/token |

Byte-level BPE over char-level because a char tokenizer needs ~4x more tokens
per story — the same 256-token context would see a quarter as much text. The
byte-level base alphabet also means any UTF-8 text is representable, which keeps
the pipeline language-agnostic for a future Tagalog run.

## Run it

```bash
pip install -e .

python test_munti.py                        # the gate — must pass first
python -m munti.data prepare --limit 4000   # dev slice (omit --limit for the real thing)
python -m munti.train --config configs/tiny.yaml
python -m munti.sample --ckpt out-tiny/ckpt.pt --prompt "Once upon a time"
```

`data/` is gitignored — regenerate it with `munti.data prepare`. Configs and
seeds are committed, so a run reproduces.

## The correctness gate

The single most useful thing in this repo. Before spending GPU hours,
`test_munti.py` proves the code is right on CPU in under a minute:

```
- test_shapes_and_init_loss   init loss 4.875 (~ln(vocab) = 4.852)
- test_causal_mask            no future leakage
- test_fast_matches_manual    fused attention == reference implementation
- test_no_pos_is_order_blind  the ablation config is genuinely order-blind
- test_overfit_batches        OVERFIT GATE: mean loss 0.0000, greedy recall 100.0%
```

The last one is the real test: deliberately memorize four fixed batches. If loss
doesn't collapse to ~0 and the model can't parrot them back, something is
miswired and no amount of training will fix it. The causal-mask test guards the
nastiest silent bug in a transformer — a leaky mask makes training loss look
*better* while generation stays gibberish.

## Can do / can't do

*Filled in with real samples after the full training run — not before.*

## License

MIT.
