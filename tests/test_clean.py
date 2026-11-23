import gzip
import json
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pipeline.clean import clean_documents, iter_jsonl_texts, iter_parquet_texts, main
from pipeline.dedup import MinHasher


def test_clean_documents_filters_redacts_and_dedups(good_paragraph):
    # Arrange
    with_email = good_paragraph + " Write to anna@example.com for details."
    texts = [with_email, with_email.upper().lower(), "too short", good_paragraph]
    stats: Counter[str] = Counter()

    # Act
    output = list(clean_documents(texts, MinHasher(), stats))

    # Assert
    assert len(output) == 1
    assert "[EMAIL]" in output[0]
    assert stats["input"] == 4
    assert stats["rejected_quality"] == 1
    assert stats["unique_exact"] == 2
    assert stats["output"] == 1


def test_iter_jsonl_texts_fails_fast_with_line_number(tmp_path):
    shard = tmp_path / "bad.jsonl"
    shard.write_text('{"text": "ok"}\n{"title": "no text"}\n')

    with pytest.raises(ValueError, match=r"bad.jsonl:2"):
        list(iter_jsonl_texts(shard))


def test_main_writes_output_and_report(tmp_path, good_paragraph):
    shard = tmp_path / "raw.jsonl"
    shard.write_text(json.dumps({"text": good_paragraph}) + "\n")
    output = tmp_path / "clean" / "out.jsonl"
    report = tmp_path / "clean" / "report.json"

    main(["--input", str(shard), "--output", str(output), "--report", str(report)])

    assert json.loads(output.read_text())["text"] == good_paragraph
    assert json.loads(report.read_text())["output"] == 1
    assert not output.with_suffix(".jsonl.partial").exists()


def test_main_rejects_missing_input(tmp_path):
    with pytest.raises(FileNotFoundError):
        main(["--input", str(tmp_path / "nope.jsonl"), "--output", "o", "--report", "r"])


def test_main_reads_parquet_and_writes_gzip(tmp_path, good_paragraph):
    # Arrange
    shard = tmp_path / "raw.parquet"
    pq.write_table(pa.table({"text": [good_paragraph, good_paragraph, "too short"]}), shard)
    output = tmp_path / "clean.jsonl.gz"
    report = tmp_path / "report.json"

    # Act
    main(["--input", str(shard), "--output", str(output), "--report", str(report)])

    # Assert
    with gzip.open(output, "rt", encoding="utf-8") as handle:
        texts = [json.loads(line)["text"] for line in handle]
    assert texts == [good_paragraph]
    assert json.loads(report.read_text())["input"] == 3


def test_iter_parquet_texts_requires_text_column(tmp_path):
    shard = tmp_path / "no_text.parquet"
    pq.write_table(pa.table({"body": ["x"]}), shard)

    with pytest.raises(ValueError, match="no 'text' column"):
        list(iter_parquet_texts(shard))
