"""Turn the cleaned corpus into flat binary token files for pretraining.

    python -m bpe.encode_corpus --tokenizer bpe/tinygpt-bpe-8k.json \
        --input data/clean/fineweb-edu-000.jsonl.gz --output-dir data/tokens

Writes train.bin and val.bin: every document's token ids followed by
<|endoftext|>, as little-endian uint16 (the vocabulary is far below 65,536, so
2 bytes per token). Training reads them with numpy.memmap, so a multi-GB corpus
never has to fit in RAM. meta.json records what produced the files.
"""

import argparse
import hashlib
import json
import multiprocessing
import os
import sys
import time
from collections.abc import Iterable, Iterator
from itertools import batched
from pathlib import Path

import numpy as np

from bpe.tokenizer import END_OF_TEXT, ByteLevelBPE
from bpe.train import iter_clean_documents

TOKEN_DTYPE = np.dtype("<u2")
# Every 200th document goes to validation: a 0.5% split spread across the whole
# shard, so validation loss reflects the same mix of sources as training.
VALIDATION_EVERY = 200
_BATCH_DOCUMENTS = 2_048
_PROGRESS_EVERY_DOCUMENTS = 50_000

_worker_tokenizer: ByteLevelBPE | None = None


def _init_worker(tokenizer_path: Path) -> None:
    global _worker_tokenizer
    _worker_tokenizer = ByteLevelBPE.load(tokenizer_path)


def _encode_in_worker(text: str) -> list[int]:
    if _worker_tokenizer is None:
        raise RuntimeError("Worker was started without _init_worker")
    return _worker_tokenizer.encode(text)


def _encode_all(texts: Iterable[str], tokenizer_path: Path, workers: int) -> Iterator[list[int]]:
    if workers == 1:
        _init_worker(tokenizer_path)
        yield from map(_encode_in_worker, texts)
        return
    context = multiprocessing.get_context("spawn")
    with context.Pool(workers, initializer=_init_worker, initargs=(tokenizer_path,)) as pool:
        for batch in batched(texts, _BATCH_DOCUMENTS):
            yield from pool.map(
                _encode_in_worker, batch, chunksize=max(1, len(batch) // (workers * 4))
            )


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def encode_corpus(
    tokenizer_path: Path, input_path: Path, output_dir: Path, workers: int = 1
) -> dict[str, object]:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    tokenizer = ByteLevelBPE.load(tokenizer_path)
    if tokenizer.vocab_size > np.iinfo(TOKEN_DTYPE).max + 1:
        raise ValueError(f"vocab_size {tokenizer.vocab_size} does not fit in {TOKEN_DTYPE}")
    end_of_text = tokenizer.special_id(END_OF_TEXT)

    output_dir.mkdir(parents=True, exist_ok=True)
    final_paths = {"train": output_dir / "train.bin", "val": output_dir / "val.bin"}
    partial_paths = {
        name: path.with_name(path.name + ".partial") for name, path in final_paths.items()
    }
    tokens = {"train": 0, "val": 0}
    documents = {"train": 0, "val": 0}
    started = time.monotonic()

    handles = {name: path.open("wb") for name, path in partial_paths.items()}
    try:
        texts = iter_clean_documents(input_path)
        for index, ids in enumerate(_encode_all(texts, tokenizer_path, workers)):
            split = "val" if index % VALIDATION_EVERY == 0 else "train"
            np.asarray([*ids, end_of_text], dtype=TOKEN_DTYPE).tofile(handles[split])
            tokens[split] += len(ids) + 1
            documents[split] += 1
            if (index + 1) % _PROGRESS_EVERY_DOCUMENTS == 0:
                total = tokens["train"] + tokens["val"]
                print(f"{index + 1:,} documents, {total:,} tokens", flush=True)
    finally:
        for handle in handles.values():
            handle.close()

    for name in final_paths:
        partial_paths[name].replace(final_paths[name])

    meta: dict[str, object] = {
        "dtype": TOKEN_DTYPE.str,
        "vocab_size": tokenizer.vocab_size,
        "end_of_text_id": end_of_text,
        "validation_every": VALIDATION_EVERY,
        "tokens": tokens,
        "documents": documents,
        "tokenizer_sha256": sha256_of_file(tokenizer_path),
        "source": input_path.name,
        "seconds": round(time.monotonic() - started, 1),
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args(argv)
    missing = [str(p) for p in (args.tokenizer, args.input) if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Inputs not found: {', '.join(missing)}")
    meta = encode_corpus(args.tokenizer, args.input, args.output_dir, args.workers)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
