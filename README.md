# tinygpt

A small decoder-only language model that I build and train from scratch, then serve
as the assistant on [masayeed.co.uk](https://www.masayeed.co.uk). It covers the whole
LLM lifecycle using free tiers only: Google Colab for training, Railway for serving and
Supabase for storage.

| Phase | Stage | Status |
|---|---|---|
| 1 | **Data curation:** normalisation, Gopher-style quality filtering, PII redaction, exact + MinHash/LSH near-dedup, data card | done |
| 2 | **Tokenizer:** byte-level BPE written from scratch, 8,192 tokens, 3.92 bytes/token on held-out text | done |
| 3 | **Model:** 17.3M-param decoder (RMSNorm, RoPE, SwiGLU, tied embeddings, KV cache); corpus encoded to 880M tokens | done |
| 4 | **Pretraining:** Colab T4, mixed precision, warmup + cosine LR, resumable checkpoints | planned |
| 5 | **Supervised fine-tuning:** chat template, assistant-only loss masking | planned |
| 6 | **Evaluation:** bits-per-byte, held-out Q&A, red-team suite, model card | planned |
| 7 | **Serving:** safetensors export, NumPy inference engine, FastAPI + SSE on Railway | planned |
| 8 | **Storage:** Supabase with row-level security, rate limits, 30-day log retention | planned |
| 9 | **Chat widget** on the portfolio site | planned |

`lesson01/` is where this project started: a character-level GPT trained on Shakespeare.

## Phase 1: cleaning the data

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements/dev.txt
.venv/bin/python -m pytest

.venv/bin/python -m pipeline.download          # pinned, sha256-checked FineWeb-Edu shard
.venv/bin/python -m pipeline.sample --input data/raw/fineweb-edu-000_00000.parquet \
    --output data/raw/sample-50k.jsonl --documents 50000
.venv/bin/python -m pipeline.clean --input data/raw/sample-50k.jsonl \
    --output data/clean/sample-50k.jsonl --report data/clean/sample-50k.report.json
.venv/bin/python -m pipeline.analyze --raw data/raw/sample-50k.jsonl \
    --clean data/clean/sample-50k.jsonl --report data/clean/sample-50k.report.json \
    --output reports/phase1-sample.md
```

Sources, licences and measured results are in [DATA_CARD.md](DATA_CARD.md), with the
sample analysis in [reports/phase1-sample.md](reports/phase1-sample.md).

Order of the pipeline: normalise (NFKC, strip invisible characters) → quality filter →
redact emails, phone numbers and IPs → exact dedup (sha256) → near dedup (MinHash, 128
permutations, 16 LSH bands, Jaccard ≥ 0.8). The report counts every rejection rule, so
each filter's effect can be measured.

## Phase 2: the tokenizer

```bash
.venv/bin/python -m bpe.train --input data/clean/fineweb-edu-000.jsonl.gz \
    --documents 100000 --vocab-size 8192 --output bpe/tinygpt-bpe-8k.json
.venv/bin/python -m bpe.evaluate --tokenizer bpe/tinygpt-bpe-8k.json \
    --clean data/clean/fineweb-edu-000.jsonl.gz --raw-dir data/raw --output reports/tokenizer-8k.md
```

Byte-level BPE with a GPT-2-style pre-tokenizer, trained in 54 s on 100k documents
(95.7M pre-tokens). Training updates pair counts incrementally and uses a heap with lazy
deletion; a test checks it learns exactly the same merges as textbook BPE. Results are in
[reports/tokenizer-8k.md](reports/tokenizer-8k.md), and the input format is in
[docs/chat_template.md](docs/chat_template.md). Plain `encode()` can never produce the
role tokens, so users cannot spoof `<|system|>` or `<|assistant|>`.

## Phase 3: the model

`model/gpt.py`: a Llama-style decoder with RMSNorm, rotary position embeddings, a SwiGLU
feed-forward, tied embeddings and a preallocated KV cache. The default config has
17.3M parameters (8 layers, width 384, 6 heads, context 512). Tests check the parameter
count, causality (future tokens can't change past logits), that KV-cache decoding equals
a full forward pass, initial loss ≈ ln(vocab), loss masking, and that the model can
memorise one batch.

```bash
.venv/bin/python -m bpe.encode_corpus --tokenizer bpe/tinygpt-bpe-8k.json \
    --input data/clean/fineweb-edu-000.jsonl.gz --output-dir data/tokens
```

This writes `train.bin` (880.4M tokens) and `val.bin` (4.3M tokens, every 200th document)
as uint16, ready for memory-mapped training. See
[reports/tokens-meta.json](reports/tokens-meta.json).
