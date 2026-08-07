# PRD — Munti: A Tiny LLM Built From Scratch

**Status:** draft, not started · **Name:** Munti (Filipino: *tiny*) ·
**Est. effort:** ~2–3 weeks part-time · **Cost:** ₱0 (free tier only — hard req)

---

## 1. Problem / motivation

Most AI portfolios show that someone can *use* a language model (an API call, a
fine-tune, a RAG app). Very few show that the person understands how an LLM
*works* — the transformer, the training loop, the tokenizer — by building one
from the ground up. Munti is that proof: a small language model implemented and
trained from scratch, validated to produce coherent text.

It is deliberately **tiny** so it trains on free hardware in hours, and it is
honestly framed as a **learning artifact**, not a product.

## 2. What Munti is

A ~10–35M parameter, decoder-only transformer (nanoGPT-class), written in pure
PyTorch and trained from scratch on **TinyStories** (a synthetic dataset of
simple children's stories purpose-built so small models produce readable
output). Given a prompt, it continues the text one token at a time.

The portfolio thesis: **I can build and train a transformer from scratch, and I
can tell you exactly what a 20M-parameter model can and cannot do.**

## 3. Goals

- Implement the core from scratch: tokenizer, transformer blocks (attention +
  MLP), training loop, and sampling — in readable, annotated PyTorch.
- Train on TinyStories on a **free** GPU (Kaggle T4) to coherent output.
- Produce an **honest capability writeup**: loss curves, sample generations, and
  a clear "can do / can't do" section.
- Ship a portfolio card + case study, optionally with a live in-browser demo.

### Non-goals

- Not a useful chatbot or assistant — it can't answer questions, hold a
  conversation, or recall facts. Say so plainly.
- No paid compute, no multi-GPU, no models beyond what a single free T4 trains
  in hours.
- **Tagalog is a separate future project** that reuses this codebase — out of
  scope here. Keep the code language-agnostic so that reuse is trivial.
- No fine-tuning of pretrained weights — that would defeat "from scratch."

## 4. Users

| user | need |
|---|---|
| Primary — recruiter / ML interviewer | Evidence the candidate understands transformer internals and reports limits honestly |
| Secondary — the curious visitor | A readable "here's a language model I built, watch it write" demo |

## 5. Functional requirements

### 5.1 Data
- **FR-1** Use the TinyStories dataset (free, Hugging Face). Cache locally.
- **FR-2** A small prepared validation split held out for loss tracking.

### 5.2 Tokenizer (from scratch counts here)
- **FR-3** Build the tokenizer, not just import one: char-level (simplest) or a
  small byte-level BPE via the `tokenizers` library. Document the choice and
  vocab size.

### 5.3 Model
- **FR-4** Decoder-only transformer in pure PyTorch: token + positional
  embeddings, N transformer blocks (multi-head causal self-attention + MLP,
  residual + layernorm), a final projection to vocab. ~10–35M params.
- **FR-5** Config-driven (layers, heads, embedding dim, context length) so
  ablations are one flag each.
- **FR-6** Training loop from scratch: batching, cross-entropy next-token loss,
  AdamW, LR warmup+decay, gradient clipping, checkpointing.
- **FR-7** Sampling from scratch: temperature + top-k, prompt → continuation.

### 5.4 Evaluation & artifacts
- **FR-8** Log train/val loss over steps; save the curve.
- **FR-9** At least one small **ablation** (e.g. with vs. without positional
  encoding, or 2 vs. 4 layers) with its loss impact — shows understanding, not
  just a run.
- **FR-10** Capture sample generations at several checkpoints ("what it learned
  over time") for the writeup — no live model needed to show them.

### 5.5 Product
- **FR-11** README + case study: how a transformer works (in your words), the
  build, loss curves, samples, ablation, and an honest **can-do / can't-do**
  section.
- **FR-12** *(stretch)* Export to run in-browser (ONNX / transformers.js) for a
  live "type a prompt" demo on a free static host.

## 6. Non-functional requirements

- **NFR-1** 100% free: Kaggle free GPU, free dataset, free static host. No paid
  anything.
- **NFR-2** A full training run fits in a single free Kaggle session (single-
  digit hours).
- **NFR-3** Reproducible: pinned deps, committed configs, fixed seed, the
  training notebook committed.
- **NFR-4** Language-agnostic pipeline (so the future Tagalog project reuses it
  unchanged).
- **NFR-5** From-scratch integrity: no importing a pretrained model or a
  ready-made GPT class. nanoGPT may be studied as a reference, but the code is
  reimplemented and annotated.

## 7. Success metrics

| metric | target |
|---|---|
| Coherent output | model produces grammatical, on-topic short story continuations |
| Val loss | trends down and converges; curve committed |
| Ablation | at least one, with a measured, explained effect |
| Honesty | explicit can-do / can't-do section grounded in real samples |

Ship tiers:
- **Minimum:** from-scratch model trains on TinyStories to coherent output;
  loss curve + samples + README.
- **Good:** + one ablation, checkpoint-progression samples, clean case study.
- **Headline:** + a live in-browser demo of your own from-scratch model.

## 8. Milestones

| week | deliverable | exit gate |
|---|---|---|
| 1 | Tokenizer + model + training loop; overfit a tiny batch on purpose | model can memorize a few batches (loss → ~0) — proves the code is correct |
| 2 | Full TinyStories training run on Kaggle T4; loss curve; sampling | coherent story continuations from held-out prompts |
| 3 | Ablation, case study/README, portfolio card; optional browser demo | honest writeup published with curves + samples |

## 9. Risks

| risk | mitigation |
|---|---|
| Output is gibberish (can't tell bug vs. under-training) | Week-1 "overfit one batch" test isolates code correctness before the big run; TinyStories is chosen precisely because tiny models CAN succeed on it |
| Free GPU session timeout mid-run | Checkpoint often; resume from checkpoint; keep the model in the trainable-in-hours band |
| Scope creep toward "make it useful" | Non-goals are binding — it's a learning artifact; usefulness is not a goal |
| Accidentally not "from scratch" | NFR-5: reimplement, don't import a GPT class; reference nanoGPT for understanding only |

## 10. Open questions

- Tokenizer: char-level (simplest, larger sequences) vs. small BPE (fewer
  tokens, a bit more work)? Lean char-level for v1 unless BPE is easy.
- Exact size: 10M (fast, safe) vs. ~30M (better output, still free)? Decide
  after the week-1 correctness test based on observed train speed.
- Browser demo in v1 or deferred to a polish pass?

## 11. Stack

Python · PyTorch (from scratch) · Hugging Face `tokenizers` + `datasets`
(TinyStories) · Kaggle free GPU · matplotlib (loss curves) · optional
ONNX / transformers.js for the browser demo · static host (Vercel/Pages).
