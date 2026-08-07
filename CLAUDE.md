# Munti — working contract

Tiny LLM from scratch. Spec: [PRD.md](PRD.md) · plan: [PLAN.md](PLAN.md) ·
kickoff: [HANDOFF.md](HANDOFF.md).

## Hard constraints (do not violate)

1. **From scratch.** The tokenizer training, transformer, training loop, and
   sampling are written here. nanoGPT is reference-only. Never import a
   pretrained model or a ready-made GPT class. The `tokenizers` library is
   permitted (PRD FR-3) but the BPE is *trained on TinyStories here*, never
   downloaded.
2. **₱0.** Free Kaggle GPU, free dataset, free host. If something needs paid
   compute, shrink the model instead.
3. **Tiny.** 10–35M params, one free Kaggle session.
4. **Honest.** README states what it can't do, quoting real failed samples.
5. **Language-agnostic.** Byte-level BPE, no English assumptions anywhere. A
   future Tagalog project swaps the corpus and reuses this unchanged.

## Layout

`munti/` — tokenizer, data, model, train, sample. `configs/` — YAML, one per
run. `test_munti.py` — correctness gate. `notebooks/` — thin Kaggle driver that
imports `munti/` and holds no logic.

## The gate

`python test_munti.py` must pass before any GPU run. It overfits 4 fixed
batches to loss < 0.05 — if that fails, the model or loop is broken and no
amount of training will save it.

## Style

Readable over clever; this code is a teaching artifact. Annotate the *why* of
each transformer piece. Manual attention stays as the reference implementation
even when the fused path is enabled for speed.
