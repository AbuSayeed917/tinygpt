"""Learn BPE merges from the cleaned corpus.

    python -m bpe.train --input data/clean/fineweb-edu-000.jsonl.gz \
        --documents 100000 --vocab-size 8192 --output bpe/tinygpt-bpe-8k.json

Algorithm (Sennrich et al., 2016, on bytes as in GPT-2):
  1. Split every document into pre-tokens and count each distinct one.
  2. Find the most frequent adjacent pair of symbols, weighted by those counts.
  3. Replace it everywhere with a new symbol; repeat until the vocabulary is full.

Recounting every pair after each merge would be O(corpus) per merge. Instead,
only words containing the merged pair are revisited, and a heap with lazy
deletion yields the next most frequent pair.
"""

import argparse
import gzip
import heapq
import json
import multiprocessing
import os
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from itertools import batched, islice
from pathlib import Path

from bpe.tokenizer import PRETOKEN_REGEX, SPECIAL_TOKENS, ByteLevelBPE, Pair, merge_pair

_BYTE_COUNT = 256
_COUNT_BATCH_DOCUMENTS = 2_000
_PROGRESS_EVERY_MERGES = 1_000


def iter_clean_documents(path: Path, skip: int = 0, limit: int | None = None) -> Iterator[str]:
    """Yields the "text" of each line of a (gzipped) JSONL file."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        stop = None if limit is None else skip + limit
        for line_number, line in enumerate(islice(handle, skip, stop), start=skip + 1):
            try:
                yield json.loads(line)["text"]
            except (json.JSONDecodeError, KeyError) as error:
                raise ValueError(f"{path}:{line_number}: bad record ({error})") from error


def _count_batch(texts: tuple[str, ...]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(PRETOKEN_REGEX.findall(text))
    return counts


def count_pretokens(texts: Iterable[str], workers: int = 1) -> Counter[str]:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    batches = batched(texts, _COUNT_BATCH_DOCUMENTS)
    if workers == 1:
        partial_counts: Iterable[Counter[str]] = map(_count_batch, batches)
        return sum(partial_counts, Counter())

    total: Counter[str] = Counter()
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        for counts in pool.imap_unordered(_count_batch, batches):
            total.update(counts)
    return total


def learn_merges(
    word_counts: Mapping[str, int],
    num_merges: int,
    min_pair_count: int = 2,
    log_progress: bool = False,
) -> list[Pair]:
    if num_merges < 0:
        raise ValueError("num_merges must be non-negative")

    words = [list(word.encode("utf-8")) for word in word_counts]
    frequencies = list(word_counts.values())
    pair_counts: Counter[Pair] = Counter()
    words_with_pair: defaultdict[Pair, set[int]] = defaultdict(set)
    for word_index, (symbols, frequency) in enumerate(zip(words, frequencies, strict=True)):
        for pair in zip(symbols, symbols[1:], strict=False):
            pair_counts[pair] += frequency
            words_with_pair[pair].add(word_index)

    # Ties break on the smaller pair, so training is fully deterministic.
    heap = [(-count, pair) for pair, count in pair_counts.items()]
    heapq.heapify(heap)
    merges: list[Pair] = []
    started = time.monotonic()

    while len(merges) < num_merges and heap:
        negative_count, pair = heapq.heappop(heap)
        current_count = pair_counts.get(pair, 0)
        if -negative_count != current_count:
            continue  # stale entry; the up-to-date count was pushed when it changed
        if current_count < min_pair_count:
            break

        new_id = _BYTE_COUNT + len(merges)
        merges.append(pair)
        count_changes: Counter[Pair] = Counter()
        for word_index in words_with_pair.pop(pair, set()):
            old_symbols = words[word_index]
            new_symbols = merge_pair(old_symbols, pair, new_id)
            if len(new_symbols) == len(old_symbols):
                continue
            frequency = frequencies[word_index]
            for old_pair in zip(old_symbols, old_symbols[1:], strict=False):
                count_changes[old_pair] -= frequency
            for new_pair in zip(new_symbols, new_symbols[1:], strict=False):
                count_changes[new_pair] += frequency
                words_with_pair[new_pair].add(word_index)
            words[word_index] = new_symbols

        for changed_pair, change in count_changes.items():
            if change == 0:
                continue
            updated = pair_counts[changed_pair] + change
            if updated <= 0:
                pair_counts.pop(changed_pair, None)
                continue
            pair_counts[changed_pair] = updated
            heapq.heappush(heap, (-updated, changed_pair))
        pair_counts.pop(pair, None)

        if log_progress and len(merges) % _PROGRESS_EVERY_MERGES == 0:
            elapsed = time.monotonic() - started
            print(f"{len(merges):,} merges, last count {current_count:,}, {elapsed:.0f}s")
    return merges


def train_tokenizer(
    texts: Iterable[str], vocab_size: int, workers: int = 1, log_progress: bool = False
) -> ByteLevelBPE:
    num_merges = vocab_size - _BYTE_COUNT - len(SPECIAL_TOKENS)
    if num_merges < 0:
        raise ValueError(f"vocab_size must be at least {_BYTE_COUNT + len(SPECIAL_TOKENS)}")
    word_counts = count_pretokens(texts, workers)
    if log_progress:
        print(f"{len(word_counts):,} distinct pre-tokens, {sum(word_counts.values()):,} total")
    merges = learn_merges(word_counts, num_merges, log_progress=log_progress)
    return ByteLevelBPE(merges)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--documents", type=int, default=100_000)
    parser.add_argument("--vocab-size", type=int, default=8_192)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args(argv)
    if not args.input.is_file():
        raise FileNotFoundError(f"Input not found: {args.input}")

    started = time.monotonic()
    texts = iter_clean_documents(args.input, limit=args.documents)
    tokenizer = train_tokenizer(texts, args.vocab_size, args.workers, log_progress=True)
    tokenizer.save(args.output)
    print(
        f"Saved {args.output} (vocab {tokenizer.vocab_size:,}) in {time.monotonic() - started:.0f}s"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
