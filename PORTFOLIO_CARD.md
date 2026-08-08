# Portfolio card — ready to paste

Drop into `Portfolio/src/data.js`, next to the two fine-tuning cards. Munti is
the "from scratch" end of the LLM spectrum that complements them.

**Live case study:** https://zeref538.github.io/munti/ — published, matching the
sibling projects' `demo` pattern. The `demo` link below works.

**Still needed before publishing:** the `images` paths don't exist yet. Screenshot
the live case study (hero with the token strip, the loss curve, the ablation
section, the can't-do table), save as `Portfolio/public/projects/munti-*.jpg`, or
trim the array to what you actually have.

```js
  {
    title: "Munti — A Tiny LLM From Scratch",
    groups: ["From Scratch"],
    description:
      "Built a 12.5M-parameter language model from the ground up in pure PyTorch — the tokenizer, the attention, the training loop and the sampler, with no pretrained weights and no ready-made GPT class — then trained it on TinyStories for 44 minutes on a free Kaggle T4 to a val loss of 1.505, where it writes coherent multi-paragraph children's stories with working dialogue. Before spending a single GPU hour I proved the code correct by deliberately overfitting four fixed batches to zero loss, which is what separates 'my model is undertrained' from 'my causal mask is broken'. The ablation is the part I'd defend hardest: removing positional embeddings cost only 0.039 nats when I had predicted word salad, so instead of publishing the number I probed it — shuffling the input tokens destroys that model, proving a causal decoder recovers position from prefix length on its own.",
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
| `munti-4.jpg` | the correctness-gate terminal output (all five tests passing) |
