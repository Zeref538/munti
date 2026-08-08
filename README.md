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

**Status:** trained. 44.4 minutes on a free Kaggle T4, final val loss **1.505**.

**→ [Read the case study](https://zeref538.github.io/munti/)** ·
[markdown version](CASE_STUDY.md) ·
[weights](https://github.com/Zeref538/munti/releases)

> Once upon a time, there was a little girl named Lily. She loved playing with
> her toys, especially her dolls. One day, Lily's friend Timmy came over to play.
> "Wow, your toys look so cool!" said Timmy. "Thank you!" said Lily.

## Results

| | |
|---|---|
| params | 12,292,992 |
| corpus | 2.1M TinyStories → ~536M tokens |
| training | 20k steps × 64 × 256 (~0.6 epoch) |
| final loss | train 1.486 / **val 1.505** |
| time / cost | 44.4 min on one free T4 / ₱0 |

[loss curve](results/curve.png) · [progression samples](results/samples.md) ·
[final generations](results/final_generations.md) ·
[ablation](results/ablation/) · [order-sensitivity probe](results/eval/)

### The ablation, and the part I got wrong

Removing the learned positional embeddings cost only **~0.04 nats** and the output
stayed coherent — I had predicted word salad. Probing it showed why: shuffling the
input tokens *destroys* the no-positions model (1.543 → 9.729, worse than the
baseline's 1.495 → 9.518), so it is not order-blind at all. A causal decoder
recovers position from prefix length, because token 5 attends over 5 tokens and
token 50 over 50.

Replicated at a second seed, because one reading isn't a measurement:

| seed | baseline | no positions | gap | spread |
|---|---|---|---|---|
| 1337 | 1.5038 | 1.5504 | +0.0467 | 0.037 – 0.063 |
| 1338 | 1.5010 | 1.5409 | +0.0400 | 0.024 – 0.052 |

The two baselines land **0.003** apart while the gap is **~0.04** — between-seed
noise is an order of magnitude below the effect, so it's real but imprecise. Full
reasoning in the [case study](CASE_STUDY.md#5-the-ablation--where-i-was-wrong).

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
| compression | 3.96 chars/token |

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

### The full run (free Kaggle T4)

[notebooks/train_kaggle.ipynb](notebooks/train_kaggle.ipynb) drives it — published
at [kaggle.com/code/johnandreimartinez/munti-train](https://www.kaggle.com/code/johnandreimartinez/munti-train).

Upload the repo as a Kaggle Dataset (any name of 6+ characters — the notebook
finds the repo by looking for `pyproject.toml`), attach it to a notebook with
**GPU T4** and **internet on**, then Run All → *Save Version → Save & Run All*.
If the 9h session limit kills the run, attach that notebook's output to a new one
and it resumes at the exact step, optimizer and scaler state included.

Via the CLI, the archive flag is not optional — without it the upload silently
drops every subdirectory and reports success:

```bash
kaggle datasets create -p <staged-repo> -r zip
kaggle kernels push -p <kernel-dir> --accelerator NvidiaTeslaT4
```

The notebook contains no logic — it imports this package, so what runs on the
GPU is the code that passed the gate. Its first cell asserts the GPU is sm_70+,
because Kaggle's default P100 is unsupported by the installed PyTorch and
everything before training is CPU work, so a doomed run looks healthy for ten
minutes.

## The correctness gate

The single most useful thing in this repo. Before spending GPU hours,
`test_munti.py` proves the code is right on CPU in under a minute:

```
- test_shapes_and_init_loss   init loss 4.875 (~ln(vocab) = 4.852)
- test_causal_mask            no future leakage
- test_fast_matches_manual    fused attention == reference implementation
- test_no_pos_is_order_blind  ablation config is permutation-invariant at position 0
- test_get_batch_alignment    targets are inputs shifted by exactly one
- test_overfit_batches        OVERFIT GATE: mean loss 0.0000, greedy recall 100.0%
```

The last one is the real test: deliberately memorize four fixed batches. If loss
doesn't collapse to ~0 and the model can't parrot them back, something is
miswired and no amount of training will fix it. The causal-mask test guards the
nastiest silent bug in a transformer — a leaky mask makes training loss look
*better* while generation stays gibberish. `test_get_batch_alignment` covers the
other silent one: every other test builds its own batches, so an off-by-one in
the data path's next-token shift would train a misaligned objective with all
tests green.

Note the fourth test's name oversells it — it only checks permutation invariance
at position 0, which is much weaker than the whole sequence being order-blind.
Misreading it is exactly how I got the ablation prediction wrong; see the
[case study](CASE_STUDY.md#5-the-ablation--where-i-was-wrong).

## Can do / can't do

Real generations at temperature 0.8 — see
[results/final_generations.md](results/final_generations.md).

**Can:** write grammatical multi-paragraph children's stories; keep a character's
name and pronouns across paragraphs; produce dialogue with correct attribution;
continue a held-out prompt in style.

**Cannot:**

| asked | answered |
|---|---|
| *What is the capital of France?* | *Her Mom laughed and said "that's the container of Santa."* |
| *The mitochondria is* | *a friend for you. Come on, let's go and find the star in the dark room.* |
| `def fibonacci(n):` | *"Thank you for marrying me." Snow smiled and replied…* |

Entity tracking also breaks inside its *good* samples — *"He opened the box and
found a box!"*, *"The cat and the cat became friends."*

None of that is a defect. It saw simple children's stories and learned exactly
that distribution. A 12M-param model trained for 44 minutes on one narrow corpus
has no mechanism for facts, arithmetic or code. It demonstrates the machinery; it
is not an assistant.

## Weights

On the [releases page](../../releases) — 47MB, optimizer state stripped (the
training checkpoint is 141MB, two-thirds of which is AdamW moments needed only to
resume).

```python
import torch
from munti.model import Munti
from munti.sample import generate_text
from munti import tokenizer as tk

model = Munti.from_checkpoint(torch.load("munti-12m.pt", weights_only=False))
print(generate_text(model, tk.load(), "Once upon a time", max_new_tokens=120))
```

## License

MIT.
