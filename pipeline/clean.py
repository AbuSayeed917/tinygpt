"""Run the full cleaning pipeline over Parquet or JSONL files with a "text" field.

    python -m pipeline.clean --input data/raw/fineweb-edu-000_00000.parquet \
        --output data/clean/fineweb-edu-000.jsonl.gz --report reports/fineweb-edu-000.json

Inputs ending in .parquet are streamed in batches; an output ending in .gz is gzipped.

Order: normalise -> quality filter -> PII redaction -> exact dedup -> near dedup.
Dedup runs last so it compares the text the model will actually see.

The per-document steps (and computing both fingerprints) run in a pool of worker
processes. Dedup decisions need the memory of every document kept so far, so they
stay in the main process, which consumes results in input order: the output is
identical whatever the number of workers.
"""

import argparse
import gzip
import json
import multiprocessing
import os
import sys
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from itertools import batched
from pathlib import Path
from typing import TextIO

import numpy as np
import pyarrow.parquet as pq

from pipeline.dedup import MinHasher, NearDuplicateIndex, exact_fingerprint
from pipeline.normalize import normalize_text
from pipeline.pii import redact_pii
from pipeline.quality import assess_quality

_PROGRESS_EVERY = 50_000
# Pool.imap would read the whole input ahead of the workers; feeding fixed-size
# batches bounds memory to one batch in flight.
_BATCH_DOCUMENTS = 4_096
_MINHASH_SEED = 0


def iter_jsonl_texts(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON ({error.msg})") from error
            text = record.get("text") if isinstance(record, dict) else None
            if not isinstance(text, str):
                raise ValueError(f"{path}:{line_number}: record has no string 'text' field")
            yield text


def iter_parquet_texts(path: Path) -> Iterator[str]:
    parquet = pq.ParquetFile(path)
    if "text" not in parquet.schema_arrow.names:
        raise ValueError(f"{path} has no 'text' column")
    for batch in parquet.iter_batches(columns=["text"], batch_size=1024):
        for text in batch.column(0).to_pylist():
            if not isinstance(text, str):
                raise ValueError(f"{path}: found a non-string 'text' value")
            yield text


def iter_texts(path: Path) -> Iterator[str]:
    if path.suffix == ".parquet":
        return iter_parquet_texts(path)
    return iter_jsonl_texts(path)


def _open_for_writing(path: Path) -> TextIO:
    if path.name.endswith(".gz.partial"):
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")


@dataclass(frozen=True)
class PreparedDocument:
    """Everything the main process needs to decide on one document."""

    text: str | None
    failed_rules: tuple[str, ...] = ()
    redaction_counts: dict[str, int] = field(default_factory=dict)
    fingerprint: str = ""
    signature: np.ndarray | None = None


def prepare_document(raw_text: str, hasher: MinHasher) -> PreparedDocument:
    text = normalize_text(raw_text)
    verdict = assess_quality(text)
    if not verdict.is_kept:
        return PreparedDocument(text=None, failed_rules=verdict.failed_rules)
    redaction = redact_pii(text)
    return PreparedDocument(
        text=redaction.text,
        redaction_counts={kind: n for kind, n in redaction.counts.items() if n},
        fingerprint=exact_fingerprint(redaction.text),
        signature=hasher.signature(redaction.text),
    )


_worker_hasher: MinHasher | None = None


def _init_worker() -> None:
    global _worker_hasher
    _worker_hasher = MinHasher(seed=_MINHASH_SEED)


def _prepare_in_worker(raw_text: str) -> PreparedDocument:
    if _worker_hasher is None:
        raise RuntimeError("Worker was started without _init_worker")
    return prepare_document(raw_text, _worker_hasher)


def _prepare_all(texts: Iterable[str], workers: int) -> Iterator[PreparedDocument]:
    if workers == 1:
        hasher = MinHasher(seed=_MINHASH_SEED)
        yield from (prepare_document(text, hasher) for text in texts)
        return

    context = multiprocessing.get_context("spawn")
    with context.Pool(workers, initializer=_init_worker) as pool:
        for batch in batched(texts, _BATCH_DOCUMENTS):
            chunk_size = max(1, len(batch) // (workers * 4))
            yield from pool.map(_prepare_in_worker, batch, chunksize=chunk_size)


def clean_documents(texts: Iterable[str], stats: Counter[str], workers: int = 1) -> Iterator[str]:
    """Streams cleaned documents; `stats` is filled in as the stream is consumed."""
    if workers < 1:
        raise ValueError("workers must be at least 1")

    seen_fingerprints: set[str] = set()
    near_duplicates = NearDuplicateIndex()
    for document in _prepare_all(texts, workers):
        stats["input"] += 1
        if document.text is None or document.signature is None:
            stats["rejected_quality"] += 1
            stats.update(f"rule:{rule}" for rule in document.failed_rules)
            continue
        stats["passed_quality"] += 1
        stats.update({f"redacted:{kind}": n for kind, n in document.redaction_counts.items()})

        if document.fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(document.fingerprint)
        stats["unique_exact"] += 1

        if near_duplicates.add_unless_duplicate(document.signature):
            continue
        stats["output"] += 1
        yield document.text


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="worker processes (default: all cores but one)",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    args = _parse_args(argv)
    missing = [str(path) for path in args.input if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Input files not found: {', '.join(missing)}")

    stats: Counter[str] = Counter()
    all_texts = (text for path in args.input for text in iter_texts(path))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file and rename, so a crash never leaves a half-written shard
    # that a later step would mistake for a finished one.
    partial_output = args.output.with_name(args.output.name + ".partial")
    with _open_for_writing(partial_output) as handle:
        for text in clean_documents(all_texts, stats, args.workers):
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            if stats["output"] % _PROGRESS_EVERY == 0:
                print(f"{stats['input']:,} read, {stats['output']:,} kept", flush=True)
    partial_output.replace(args.output)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(dict(sorted(stats.items())), indent=2) + "\n")
    print(json.dumps(dict(sorted(stats.items())), indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
