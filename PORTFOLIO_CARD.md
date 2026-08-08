# Munti — portfolio integration brief

*Paste this whole file into a session working in the `Portfolio` repo. It is
self-contained: what the project is, the verified numbers, and the exact edits.*

## What Munti is

A ~12.5M-parameter language model **built from scratch** in pure PyTorch — the
byte-level BPE tokenizer, the causal multi-head attention, the training loop and
the sampler are all hand-written. No pretrained weights, no ready-made GPT class.
Trained on TinyStories on a free Kaggle T4. It writes coherent children's stories
and can do nothing else, which the project states openly.

It is the **from-scratch** end of the owner's LLM work, complementing the two
existing fine-tuning cards (Refusal Calibration, Token-Optimization).

| fact | value |
|---|---|
| params | 12,292,992 (6 layers, 6 heads, 384 wide, 256 context) |
| tokenizer | byte-level BPE, 4096 vocab, 3.96 chars/token, trained on the corpus |
| corpus | 2.1M TinyStories → ~536M tokens |
| training | 20k steps × 64 × 256 (~0.6 epoch), **44.4 min**, one free T4, ₱0 |
| result | train 1.486 / **val 1.505** (perplexity 4.50) |
| ablation | no positional embeddings → **~0.04 nats** worse, replicated at 2 seeds |
| honesty | can-do/can't-do quotes real failures, e.g. it answers "capital of France?" with a story about Santa |

The three things that make it portfolio-worthy, in order:

1. **A correctness gate before any GPU spend.** Six CPU tests, under a minute;
   the key one deliberately overfits four fixed batches to loss 0.0000 with 100%
   greedy recall. It means no GPU hour was ever spent debugging the model.
2. **An ablation whose prediction was wrong, chased down rather than buried.**
   Removing positional embeddings barely hurt. Two follow-ups explain it:
   shuffling tokens destroys that model (so it isn't order-blind — a causal
   decoder recovers position from prefix length), and a second seed reproduced
   the gap while the two baselines landed 0.003 apart.
3. **Honest limits quoted verbatim**, including failures inside its good samples.

**Links:** repo https://github.com/Zeref538/munti · live case study
https://zeref538.github.io/munti/ · weights on the GitHub releases page.

---

## What to paste where

**This file is not what goes in the codebase.** The block below is a JavaScript
object; it goes into `Portfolio/src/data.js`, in the `projects` array, next to
the two fine-tuning cards.

## Two edits, not one

`Portfolio/src/App.jsx:233` hardcodes the filter tabs:

```js
const projGroups = ["Agentic AI", "RAG", "Fine-Tuning", "ML & Forecasting", "Full-Stack"];
```

`App.jsx:237` filters with `p.groups?.includes(projFilter)`, so a card whose
group is missing from that array never appears under any tab. Pick one:

- **Add the group** — change line 233 to include `"From Scratch"`. Accurate, and
  it distinguishes Munti from the two fine-tuning projects, which is the whole
  point of having it.
- **Reuse an existing group** — set `groups: ["Fine-Tuning"]` in the card below.
  No App.jsx change, but it files a from-scratch project under fine-tuning, which
  is the opposite of what it is.

**Still needed:** the `images` paths don't exist yet. Screenshot the live case
study, save as `Portfolio/public/projects/munti-*.jpg`, or trim the array to what
you actually have.

```js
  {
    title: "Munti — A Tiny LLM From Scratch",
    groups: ["From Scratch"],
    description:
      "Built a 12.5M-parameter language model from the ground up in pure PyTorch — the tokenizer, the attention, the training loop and the sampler, with no pretrained weights and no ready-made GPT class — then trained it on TinyStories for 44 minutes on a free Kaggle T4 to a val loss of 1.505, where it writes coherent multi-paragraph children's stories with working dialogue. Before spending a single GPU hour I proved the code correct by deliberately overfitting four fixed batches to zero loss, which is what separates 'my model is undertrained' from 'my causal mask is broken'. The ablation is the part I'd defend hardest: removing positional embeddings cost only ~0.04 nats when I had predicted word salad, so instead of publishing the number I probed it twice — shuffling the input tokens destroys that model, proving a causal decoder recovers position from prefix length on its own, and a second seed reproduced the gap while the two baselines landed 0.003 apart.",
    tags: ["PyTorch", "Transformers", "From Scratch", "BPE Tokenizer", "Kaggle", "Python"],
    metric: "12.5M params from scratch · val 1.505 in 44 min",
    category: "From Scratch · Transformers · Evaluation",
    date: "2026",
    image: "/projects/munti-1.jpg",
    images: [
      "/projects/munti-1.jpg",
      "/projects/munti-2.jpg",
      "/projects/munti-3.jpg",
      "/projects/munti-4.jpg",
    ],
    link: "https://github.com/Zeref538/munti",
    demo: "https://zeref538.github.io/munti/",
    demoLabel: "case study",
    highlights: [
      "Wrote the transformer by hand — multi-head causal attention, pre-LN residual blocks, weight-tied embeddings — and proved it correct on CPU in under a minute before touching a GPU, by overfitting four fixed batches to loss 0.0000 with 100% greedy recall",
      "Predicted that removing positional embeddings would produce word salad and was wrong — it cost ~0.04 nats — so I measured why: shuffling tokens sends that model from 1.543 to 9.729, a larger collapse than the baseline's, proving causal masking already leaks position via prefix length",
      "Replicated the ablation at a second seed rather than trusting one reading: the two baselines land 0.003 apart while the effect is ~0.04, so between-seed noise sits an order of magnitude below the result being claimed",
      "Honest can-do/can't-do grounded in real generations: it writes stories with consistent characters and dialogue, but answers \"What is the capital of France?\" with a story about Santa, and loses entity tracking even mid-success (\"He opened the box and found a box!\")",
    ],
  },
```

## Suggested screenshots

| file | what |
|---|---|
| `munti-1.jpg` | `results/curve.png` — the loss curve |
| `munti-2.jpg` | a final generation from `results/final_generations.md` |
| `munti-3.jpg` | the baseline-vs-ablation table, or both curves side by side |
| `munti-4.jpg` | the correctness-gate terminal output (all six tests passing) |
