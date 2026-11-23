"""Exploratory data analysis: what the corpus looks like before and after cleaning.

    python -m pipeline.analyze --raw data/raw/sample-50k.jsonl \
        --clean data/clean/sample-50k.jsonl --report data/clean/sample-50k.report.json \
        --output reports/phase1-sample.md

Writes a Markdown report (committed to the repo) with length distributions,
quality-signal percentiles, the top source domains and each filter's effect.
"""

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from pipeline.quality import compute_quality_stats

PERCENTILES = (5, 25, 50, 75, 95)
_SIGNALS = (
    "mean_word_length",
    "alphabetic_word_fraction",
    "duplicate_line_fraction",
    "top_bigram_char_fraction",
)


def _iter_records(path: Path) -> Iterator[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON ({error.msg})") from error


def percentile_row(name: str, values: list[float]) -> str:
    if not values:
        return f"| {name} | " + " | ".join("–" for _ in PERCENTILES) + " |"
    points = np.percentile(np.asarray(values, dtype=float), PERCENTILES)
    return f"| {name} | " + " | ".join(f"{point:,.2f}" for point in points) + " |"


def _domain(url: object) -> str:
    if not isinstance(url, str):
        return "(unknown)"
    host = urlparse(url).hostname or "(unknown)"
    return host.removeprefix("www.")


def describe_raw(path: Path) -> dict[str, object]:
    characters: list[float] = []
    tokens: list[float] = []
    signals: dict[str, list[float]] = {name: [] for name in _SIGNALS}
    domains: Counter[str] = Counter()
    for record in _iter_records(path):
        text = str(record["text"])
        characters.append(len(text))
        if isinstance(record.get("token_count"), int):
            tokens.append(record["token_count"])
        stats = compute_quality_stats(text)
        for name in _SIGNALS:
            if name in stats:
                signals[name].append(stats[name])
        domains[_domain(record.get("url"))] += 1
    return {"characters": characters, "tokens": tokens, "signals": signals, "domains": domains}


def build_report(raw: dict, clean_characters: list[float], filter_stats: dict[str, int]) -> str:
    input_count = filter_stats.get("input", 0)
    output_count = filter_stats.get("output", 0)
    kept_share = output_count / input_count if input_count else 0.0
    header = "| measure | " + " | ".join(f"p{p}" for p in PERCENTILES) + " |"
    divider = "|---" * (len(PERCENTILES) + 1) + "|"

    lines = [
        "# Phase 1 sample report",
        "",
        f"Documents in: **{input_count:,}**, kept: **{output_count:,}** ({kept_share:.1%}).",
        f"Characters in: {sum(raw['characters']):,.0f}, kept: {sum(clean_characters):,.0f}.",
        "",
        "## Lengths",
        "",
        header,
        divider,
        percentile_row("raw characters", raw["characters"]),
        percentile_row("clean characters", clean_characters),
        percentile_row("GPT-2 tokens (raw, from source)", raw["tokens"]),
        "",
        "## Quality signals (raw)",
        "",
        header,
        divider,
        *(percentile_row(name, values) for name, values in raw["signals"].items()),
        "",
        "## What each filter did",
        "",
        "| step | count |",
        "|---|---|",
        *(f"| {name} | {count:,} |" for name, count in sorted(filter_stats.items())),
        "",
        "## Top 15 source domains (raw)",
        "",
        "| domain | documents |",
        "|---|---|",
        *(f"| {domain} | {count:,} |" for domain, count in raw["domains"].most_common(15)),
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True, help="JSON from pipeline.clean")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    missing = [str(p) for p in (args.raw, args.clean, args.report) if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Inputs not found: {', '.join(missing)}")

    raw = describe_raw(args.raw)
    clean_characters = [float(len(str(r["text"]))) for r in _iter_records(args.clean)]
    filter_stats = json.loads(args.report.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_report(raw, clean_characters, filter_stats))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main(sys.argv[1:])
