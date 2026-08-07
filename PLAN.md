# Munti — Phased Build Plan

> Tiny LLM from scratch, trained on TinyStories. Free tier only. ~2–3 weeks.
> Full spec: [PRD.md](PRD.md). Kickoff context: [HANDOFF.md](HANDOFF.md).

## Proposed repo layout

```
data/         TinyStories download + tokenizer training + tokenized cache
model/        transformer.py (blocks, attention, model), config.py
train/        train.py (loop, checkpointing), configs/*.yaml
sample/        generate.py (temperature + top-k sampling)
eval/         loss-curve builder, checkpoint-sample capture, ablation runner
notebooks/    train_kaggle.ipynb (the free-GPU run)
web/          optional in-browser demo (onnx/transformers.js)
CLAUDE.md     working contract
README.md / CASE_STUDY.md
```

## Phase 1 — Build it, prove it's correct (week 1, CPU/local ok)

- [ ] Tokenizer: char-level (default) or small BPE via `tokenizers`; document
      vocab size. Tokenize a TinyStories slice → cached ids.
- [ ] Model from scratch: embeddings, N transformer blocks (causal multi-head
      attention + MLP, residual + layernorm), output projection. Config-driven.
- [ ] Training loop: batching, cross-entropy, AdamW, warmup+cosine decay, grad
      clip, checkpointing.
- [ ] Sampling: temperature + top-k, prompt → continuation.
- **EXIT GATE (the key test):** deliberately **overfit a handful of batches** —
  loss should drop toward ~0 and the model should parrot them back. This proves
  the code is correct *before* spending GPU hours. Do not proceed until it passes.

## Phase 2 — Train for real on free GPU (week 2)

- [ ] `notebooks/train_kaggle.ipynb`: full TinyStories run on a free T4.
- [ ] Checkpoint frequently (survive session limits); log train/val loss.
- [ ] Save the loss curve; capture sample generations at several checkpoints.
- **EXIT GATE:** held-out prompts produce **coherent, grammatical** short-story
  continuations; loss curve committed.

## Phase 3 — Understand it + ship (week 3)

- [ ] One ablation (e.g. no positional encoding, or 2 vs 4 layers) with measured
      loss effect and a one-paragraph explanation.
- [ ] Case study/README: how a transformer works (your words), the build, loss
      curves, checkpoint-progression samples, ablation, and an honest
      **can-do / can't-do** section.
- [ ] Portfolio card (house format) + add to portfolio data.
- [ ] *(stretch)* Export to ONNX/transformers.js, ship a live in-browser
      "type a prompt" demo on a free host.
- **EXIT GATE:** published case study with curves + samples; card live.

## Success tiers (from PRD)

- **Minimum:** from-scratch model → coherent TinyStories output; curve + samples
  + README.
- **Good:** + one ablation, checkpoint-progression samples, clean case study.
- **Headline:** + live in-browser demo of your own from-scratch model.

## Ponytail notes (keep it lazy)

- Study nanoGPT as the reference skeleton, but reimplement + annotate — don't
  import a prebuilt GPT class (that breaks "from scratch").
- Char-level tokenizer first; only reach for BPE if sequence length hurts.
- Start at ~10M params (fast, safe); scale toward ~30M only after the week-1
  correctness test shows training speed allows it in a free session.
- Keep every module language-agnostic so the future Tagalog project reuses this
  codebase unchanged — no English-specific assumptions in the tokenizer/model.
```
