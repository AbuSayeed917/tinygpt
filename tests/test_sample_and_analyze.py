import json
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pipeline.analyze import build_report, percentile_row
from pipeline.sample import evenly_spaced_indices, write_sample


def test_evenly_spaced_indices_spread_across_range():
    assert evenly_spaced_indices(10, 3) == [0, 3, 6]
    assert evenly_spaced_indices(3, 5) == [0, 1, 2]


def test_evenly_spaced_indices_rejects_non_positive():
    with pytest.raises(ValueError):
        evenly_spaced_indices(0, 1)


def test_write_sample_takes_rows_from_spread_out_row_groups(tmp_path):
    # Arrange: 10 row groups of 2 rows each
    rows = {
        "id": [f"doc{i}" for i in range(20)],
        "text": [f"text {i}" for i in range(20)],
        "url": ["https://www.example.com/a"] * 20,
        "token_count": list(range(20)),
        "score": [3.0] * 20,
    }
    shard = tmp_path / "shard.parquet"
    pq.write_table(pa.table(rows), shard, row_group_size=2)
    output = tmp_path / "sample.jsonl"

    # Act
    written = write_sample(shard, output, documents=4)

    # Assert
    ids = [json.loads(line)["id"] for line in output.read_text().splitlines()]
    assert written == 4
    assert ids == ["doc0", "doc1", "doc10", "doc11"]


def test_percentile_row_handles_empty_values():
    assert percentile_row("x", []).count("–") == 5


def test_build_report_includes_kept_share_and_domains():
    raw = {
        "characters": [100.0, 200.0],
        "tokens": [20.0, 40.0],
        "signals": {"mean_word_length": [4.0, 5.0]},
        "domains": Counter({"example.com": 2}),
    }

    report = build_report(raw, [100.0], {"input": 2, "output": 1})

    assert "kept: **1** (50.0%)" in report
    assert "| example.com | 2 |" in report
