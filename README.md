# tinygpt — build an LLM from scratch, then evolve it

Each lesson is a folder that builds on the previous one. Every lesson runs on a
MacBook (M-series GPU via PyTorch "mps").

| # | Lesson | What you learn | Maps to real LLMs |
|---|--------|----------------|-------------------|
| 01 | **Char-level GPT** (done) | tokens, embeddings, attention, residuals, cross-entropy, AdamW, sampling | The 2018-2019 GPT-1/GPT-2 architecture |
| 02 | **Experiments** | break the model on purpose: no mask, no positions, 1 vs 8 layers, temperature | Why every part exists |
| 03 | **BPE tokenizer** | write byte-pair encoding yourself | GPT/Claude tokenizers |
| 04 | **Modern architecture** | RMSNorm, RoPE, SwiGLU, GQA, FlashAttention, KV cache | Llama 3 / DeepSeek / Qwen style |
| 05 | **Training like a lab** | warmup + cosine LR, gradient clipping, mixed precision, scaling laws | How pretraining runs are tuned |
| 06 | **Mixture of Experts** | router + experts, load balancing | DeepSeek-V3, Mixtral, (reportedly) frontier models |
| 07 | **SFT: make it a chatbot** | chat template, special tokens, loss masking | Instruction tuning |
| 08 | **Preference tuning** | DPO from chosen/rejected pairs | RLHF / DPO |
| 09 | **Reasoning RL** | GRPO with a verifiable reward (arithmetic) | "Thinking" models |
| 10 | **Agent loop** | your model calls a calculator tool in a loop | Agents / tool use |

## Lesson 01 quickstart

```bash
cd lesson01
python3 tokenizer.py          # see text -> numbers -> text
python3 train.py              # ~5-10 min, saves checkpoint.pt
python3 sample.py --prompt "ROMEO:" --temperature 0.8
```

## Reading what training prints

- **Starting loss ≈ ln(vocab_size)** (ln 65 ≈ 4.17): the model is guessing uniformly at random.
- **~2.5**: it has learned letter frequencies and common pairs.
- **~1.5**: it has learned words, line breaks, speaker names.
- **train falling, val rising** = overfitting (memorising, not learning).

## Files in lesson01

- `tokenizer.py` — text <-> ids
- `model.py` — the GPT (attention, MLP, blocks, generate)
- `train.py` — the training loop
- `sample.py` — generate text from the checkpoint
