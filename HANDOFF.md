# Munti — Handoff / Kickoff Context

> Copy this whole folder into a fresh empty repo folder, open a new session
> there, and paste the "Kickoff prompt" below. This file is the context a fresh
> agent needs to start building with zero prior conversation.

## What this is

**Munti** (Filipino: *tiny*) — a small language model built **from scratch** in
pure PyTorch and trained on **TinyStories** on free hardware, until it produces
coherent short-story text. Full spec in [PRD.md](PRD.md); phased build in
[PLAN.md](PLAN.md).

One-sentence identity: **proof I understand transformers by building and
training one from the ground up — and can honestly say what a 20M-parameter
model can and can't do.**

## Who it's for / who's building it

- Owner: John Andrei Martinez (GitHub `Zeref538`), AI/ML student, portfolio at
  johnandrei.vercel.app. This becomes a portfolio project card + case study.
- It is a **learning artifact, not a product.** It will NOT be a useful chatbot.
  That's fine and stated openly — the value is demonstrated understanding +
  honest evaluation.

## Non-negotiable constraints (hard requirements)

1. **From scratch.** Write the tokenizer, transformer, training loop, and
   sampling yourself. nanoGPT may be studied as a reference, but reimplement and
   annotate — do NOT import a pretrained model or a ready-made GPT class. That
   would defeat the entire point.
2. **₱0 cost.** Free Kaggle GPU, free dataset (TinyStories via Hugging Face),
   free static host. No paid compute/data. If something seems to need paid GPU,
   stop and shrink the model.
3. **Tiny + trainable in hours.** ~10–35M params, fits one free Kaggle session.
4. **Honest framing.** README has an explicit can-do / can't-do section grounded
   in real samples. No overselling.
5. **Language-agnostic code.** A separate future project will train this SAME
   codebase on Tagalog — so no English-specific assumptions baked into the
   tokenizer or model. Keep reuse trivial.

## The most important early step

**Prove the code is correct before spending GPU hours.** Phase 1's exit gate is
to deliberately *overfit a few batches* — if loss drops toward zero and the
model parrots them back, the transformer + training loop are wired correctly.
Only then run the full TinyStories training. This single test prevents the
classic "output is gibberish and I can't tell if it's a bug or under-training"
trap.

## Kickoff prompt (paste into the fresh session)

> I'm building Munti, a tiny language model from scratch (pure PyTorch) trained
> on TinyStories, on free hardware. Read PRD.md, PLAN.md, and HANDOFF.md in this
> folder — they're the full spec. Hard constraints: from scratch (write the
> tokenizer, transformer, training loop, sampling myself — no pretrained models,
> no prebuilt GPT class; nanoGPT is reference-only), free tier only (Kaggle GPU +
> TinyStories + free host), ~10–35M params, honest can-do/can't-do framing, and
> keep the code language-agnostic for a future Tagalog reuse. Start with Phase 1:
> build the tokenizer, model, and training loop, and prove correctness by
> overfitting a few batches BEFORE any full training run. Show me the module
> layout and the Phase 1 plan before writing the model code.

## First moves for the new agent

1. `git init`, `CLAUDE.md` working contract, `pyproject.toml`, folder layout
   from PLAN.md.
2. Build tokenizer → model → training loop → sampling.
3. Run the **overfit-a-batch correctness test** (Phase 1 gate) before touching
   the free GPU.
4. Only then: full TinyStories run on Kaggle, loss curve, samples, ablation,
   case study.

## Definition of done (v1)

A from-scratch model that writes coherent TinyStories-style continuations from
held-out prompts; a committed loss curve and checkpoint-progression samples; at
least one ablation; a case study/README with an honest can-do/can't-do section;
and a portfolio card in the house format (metric like
`~20M params from scratch · coherent output`).

## Related / future

- **Tagalog-from-scratch** — a separate future project that reuses this exact
  codebase on a Tagalog corpus. Not part of Munti v1; just don't block it.
- Sibling fine-tuning projects already shipped: Token-Optimization LLM
  Fine-Tuning, Refusal Calibration LLM Fine-Tuning — Munti complements them by
  covering the "from scratch" end of the LLM spectrum.
