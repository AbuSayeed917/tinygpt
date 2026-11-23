# tinygpt

A small decoder-only language model that I build and train from scratch, then serve
as the assistant on [masayeed.co.uk](https://www.masayeed.co.uk). It covers the whole
LLM lifecycle using free tiers only: Google Colab for training, Railway for serving and
Supabase for storage.

| Phase | Stage | Status |
|---|---|---|
| 1 | **Data curation:** normalisation, Gopher-style quality filtering, PII redaction, exact + MinHash/LSH near-dedup, data card | in progress |
| 2 | **Tokenizer:** byte-level BPE written from scratch | planned |
| 3 | **Model:** ~20M-param decoder (RMSNorm, RoPE, SwiGLU, tied embeddings, KV cache) | planned |
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
.venv/bin/python -m pipeline.clean --input data/raw/shard.jsonl \
    --output data/clean/shard.jsonl --report data/clean/shard.report.json
```

Order of the pipeline: normalise (NFKC, strip invisible characters) → quality filter →
redact emails, phone numbers and IPs → exact dedup (sha256) → near dedup (MinHash, 128
permutations, 16 LSH bands, Jaccard ≥ 0.8). The report counts every rejection rule, so
each filter's effect can be measured.
