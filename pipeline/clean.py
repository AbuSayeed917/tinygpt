"""Run the full cleaning pipeline over JSONL files with a "text" field.

    python -m pipeline.clean --input data/raw/shard_000.jsonl \
        --output data/clean/shard_000.jsonl --report data/clean/shard_000.report.json

Order: normalise -> quality filter -> PII redaction -> exact dedup -> near dedup.
Dedup runs last so it compares the text the model will actually see.
"""

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

from pipeline.dedup import MinHasher, drop_exact_duplicates, drop_near_duplicates
from pipeline.normalize import normalize_text
from pipeline.pii import redact_pii
from pipeline.quality import assess_quality


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


def _count_through(texts: Iterable[str], stats: Counter[str], key: str) -> Iterator[str]:
    for text in texts:
        stats[key] += 1
        yield text


def _filter_and_redact(texts: Iterable[str], stats: Counter[str]) -> Iterator[str]:
    for raw_text in texts:
        stats["input"] += 1
        text = normalize_text(raw_text)
        verdict = assess_quality(text)
        if not verdict.is_kept:
            stats["rejected_quality"] += 1
            stats.update(f"rule:{rule}" for rule in verdict.failed_rules)
            continue
        redaction = redact_pii(text)
        stats.update({f"redacted:{kind}": n for kind, n in redaction.counts.items() if n})
        yield redaction.text


def clean_documents(texts: Iterable[str], hasher: MinHasher, stats: Counter[str]) -> Iterator[str]:
    """Streams cleaned documents; `stats` is filled in as the stream is consumed."""
    passed_quality = _count_through(_filter_and_redact(texts, stats), stats, "passed_quality")
    unique_exact = _count_through(drop_exact_duplicates(passed_quality), stats, "unique_exact")
    yield from _count_through(drop_near_duplicates(unique_exact, hasher), stats, "output")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    args = _parse_args(argv)
    missing = [str(path) for path in args.input if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Input files not found: {', '.join(missing)}")

    stats: Counter[str] = Counter()
    all_texts = (text for path in args.input for text in iter_jsonl_texts(path))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file and rename, so a crash never leaves a half-written shard
    # that a later step would mistake for a finished one.
    partial_output = args.output.with_suffix(args.output.suffix + ".partial")
    with partial_output.open("w", encoding="utf-8") as handle:
        for text in clean_documents(all_texts, MinHasher(), stats):
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    partial_output.replace(args.output)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(dict(sorted(stats.items())), indent=2) + "\n")
    print(json.dumps(dict(sorted(stats.items())), indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
