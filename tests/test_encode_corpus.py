import gzip
import json

import numpy as np

from bpe.encode_corpus import VALIDATION_EVERY, encode_corpus
from bpe.tokenizer import END_OF_TEXT
from bpe.train import train_tokenizer

DOCUMENTS = [f"document number {i}: the cat sat on the mat." for i in range(VALIDATION_EVERY + 5)]


def _write_inputs(tmp_path):
    corpus = tmp_path / "clean.jsonl.gz"
    with gzip.open(corpus, "wt", encoding="utf-8") as handle:
        for text in DOCUMENTS:
            handle.write(json.dumps({"text": text}) + "\n")
    tokenizer = train_tokenizer(DOCUMENTS, vocab_size=300)
    tokenizer_path = tmp_path / "tok.json"
    tokenizer.save(tokenizer_path)
    return corpus, tokenizer, tokenizer_path


def _decode_documents(path, tokenizer):
    ids = np.fromfile(path, dtype="<u2").tolist()
    return tokenizer.decode(ids).split(END_OF_TEXT)[:-1]


def test_encode_corpus_splits_and_round_trips(tmp_path):
    # Arrange
    corpus, tokenizer, tokenizer_path = _write_inputs(tmp_path)
    output_dir = tmp_path / "tokens"

    # Act
    meta = encode_corpus(tokenizer_path, corpus, output_dir, workers=1)

    # Assert
    val_docs = _decode_documents(output_dir / "val.bin", tokenizer)
    train_docs = _decode_documents(output_dir / "train.bin", tokenizer)
    assert val_docs == [DOCUMENTS[0], DOCUMENTS[VALIDATION_EVERY]]
    assert len(train_docs) == len(DOCUMENTS) - 2
    assert meta["documents"] == {"train": len(DOCUMENTS) - 2, "val": 2}
    assert meta["tokens"]["val"] == (output_dir / "val.bin").stat().st_size // 2
    assert not list(output_dir.glob("*.partial"))


def test_parallel_encoding_matches_serial(tmp_path):
    corpus, _, tokenizer_path = _write_inputs(tmp_path)

    encode_corpus(tokenizer_path, corpus, tmp_path / "serial", workers=1)
    encode_corpus(tokenizer_path, corpus, tmp_path / "parallel", workers=2)

    for name in ("train.bin", "val.bin"):
        assert (tmp_path / "serial" / name).read_bytes() == (
            tmp_path / "parallel" / name
        ).read_bytes()
