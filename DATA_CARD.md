# Data card

Format after "Datasheets for Datasets" (Gebru et al., 2021). Each source is pinned to one
dataset commit and checked against its sha256 by `pipeline/download.py`, so the corpus can
be rebuilt exactly.

## Sources

| Name in `SOURCES` | Dataset | Use | Licence | Size |
|---|---|---|---|---|
| `fineweb-edu-0` | HuggingFaceFW/fineweb-edu, `sample/10BT/000_00000.parquet` | pretraining | ODC-By 1.0 | 726,000 docs, ≈752M GPT-2 tokens |
| `oasst2-train`, `oasst2-validation` | OpenAssistant/oasst2 | fine-tuning (chat) | Apache-2.0 | 66 MB |
| `dolly-15k` | databricks/databricks-dolly-15k | fine-tuning (instructions) | CC BY-SA 3.0 | 15k pairs |
| `squad-train`, `squad-validation` | rajpurkar/squad | fine-tuning (grounded Q&A) | CC BY-SA 4.0 | ≈98k questions |

Not used: copyrighted English-learning books. Their licences don't allow training a model
that is then published.

## Pretraining pipeline (`pipeline/clean.py`)

1. **Normalise:** NFKC, strip Unicode control/format characters, unify newlines.
2. **Quality filter:** Gopher-style heuristics (word count, mean word length, symbol
   ratio, bullet and ellipsis lines, alphabetic words, stop words, duplicate lines,
   repeated bigrams).
3. **Redact PII:** emails → `[EMAIL]`, IPv4 → `[IP]`, phone numbers → `[PHONE]`.
4. **Exact dedup:** sha256 of lower-cased, whitespace-collapsed text.
5. **Near dedup:** MinHash (5-word shingles, 128 permutations), LSH with 16 bands,
   dropped at an estimated Jaccard of 0.8 or more.

## Measured on a 50k-document sample

Full report: [`reports/phase1-sample.md`](reports/phase1-sample.md). The sample is spread
evenly across the shard's row groups.

- **Kept:** 49,693 of 50,000 (99.4%).
  - 296 failed quality, mostly `ellipsis_lines` (132) and `bullet_lines` (80).
  - 3 were exact duplicates and 4 near-duplicates.
  - FineWeb-Edu is already filtered and deduplicated upstream, so a low rejection rate is
    what we expect.
- **Redactions:** 1,482 emails, 310 IPs, 1,969 phone numbers.
- **Median document:** 2,912 characters (≈624 GPT-2 tokens).
- **Most common sources:** Wikipedia, reference.com, Britannica, eNotes, and science news
  sites.
- **Cleaning speed:** ≈440 documents per second on one CPU core (Apple M-series), so about
  27 minutes for the full shard.

## Known limitations

- **The phone pattern was tuned on samples.** The first version mostly matched ISBNs,
  PubMed IDs and page ranges (5,724 matches). The current one matches 1,971, and about 19
  of 20 sampled matches are real numbers. It still misses phone numbers written in
  unusual local formats.
- **Some PII was already masked upstream.** FineWeb masks emails (`email@example.com`) and
  IPs before release, which leaves artefacts such as fake section numbers
  ("RDA 126.96.36.199").
- **The corpus is English-only and skews US-centric** in its domains, spellings and
  examples.
- **Quality heuristics are tuned for prose.** They are not applied to the fine-tuning
  datasets, where short answers are valid. Those get their own pass in Phase 5.

## Privacy

Raw and cleaned shards are never committed (`data/raw/` and `data/clean/` are gitignored).
Only code, reports with aggregate counts, and this card are published.
