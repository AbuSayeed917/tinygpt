"""Write a JSONL sample of a Parquet shard, spread evenly across its row groups.

    python -m pipeline.sample --input data/raw/fineweb-edu-000_00000.parquet \
        --output data/raw/sample-50k.jsonl --documents 50000

Taking the first N rows would over-represent whichever crawl the shard starts
with; spacing row groups out gives a fairer picture of the whole shard.
"""

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

KEPT_COLUMNS = ("id", "text", "url", "token_count", "score")


def evenly_spaced_indices(total: int, count: int) -> list[int]:
    if total <= 0 or count <= 0:
        raise ValueError("total and count must be positive")
    if count >= total:
        return list(range(total))
    step = total / count
    return [int(i * step) for i in range(count)]


def write_sample(input_path: Path, output_path: Path, documents: int) -> int:
    if documents <= 0:
        raise ValueError("documents must be positive")
    parquet = pq.ParquetFile(input_path)
    missing = set(KEPT_COLUMNS) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"{input_path} is missing columns: {sorted(missing)}")

    group_count = parquet.metadata.num_row_groups
    rows_per_group = max(1, parquet.metadata.num_rows // group_count)
    groups_needed = -(-documents // rows_per_group)
    chosen_groups = evenly_spaced_indices(group_count, groups_needed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    written = 0
    with partial_path.open("w", encoding="utf-8") as handle:
        for group_index in chosen_groups:
            table = parquet.read_row_group(group_index, columns=list(KEPT_COLUMNS))
            for record in table.to_pylist():
                if written == documents:
                    break
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
    partial_path.replace(output_path)
    return written


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--documents", type=int, default=50_000)
    args = parser.parse_args(argv)
    if not args.input.is_file():
        raise FileNotFoundError(f"Input not found: {args.input}")
    written = write_sample(args.input, args.output, args.documents)
    print(f"Wrote {written} documents to {args.output}")


if __name__ == "__main__":
    main(sys.argv[1:])
