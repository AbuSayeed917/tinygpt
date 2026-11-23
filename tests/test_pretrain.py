import csv
import json
import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from bpe.encode_corpus import sha256_of_file
from bpe.train import train_tokenizer
from model.gpt import GPT, ModelConfig
from train.pretrain import (
    TokenFile,
    TrainConfig,
    build_optimizer,
    learning_rate,
    load_checkpoint,
    run_training,
    verify_token_files,
)

CPU = torch.device("cpu")
SCHEDULE = TrainConfig(max_steps=100, warmup_steps=10, peak_lr=1e-3, min_lr=1e-4)


@pytest.fixture
def toy_setup(tmp_path):
    tokenizer = train_tokenizer(["the cat sat on the mat"] * 5, vocab_size=270)
    tokenizer_path = tmp_path / "tok.json"
    tokenizer.save(tokenizer_path)
    data_dir = tmp_path / "tokens"
    data_dir.mkdir()
    rng = np.random.default_rng(0)
    counts = {"train": 5_000, "val": 1_000}
    for split, count in counts.items():
        rng.integers(0, tokenizer.vocab_size, count, dtype=np.uint16).tofile(
            data_dir / f"{split}.bin"
        )
    meta = {
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_sha256": sha256_of_file(tokenizer_path),
        "tokens": counts,
    }
    (data_dir / "meta.json").write_text(json.dumps(meta))
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=16,
        d_model=32,
        n_layers=2,
        n_heads=4,
        ffn_hidden=64,
    )
    train_config = TrainConfig(
        batch_size=4,
        grad_accum_steps=2,
        max_steps=6,
        warmup_steps=2,
        eval_every=3,
        eval_batches=2,
    )
    return model_config, train_config, data_dir, tokenizer_path


def test_learning_rate_warms_up_linearly():
    assert learning_rate(0, SCHEDULE) == pytest.approx(1e-4)
    assert learning_rate(9, SCHEDULE) == pytest.approx(1e-3)


def test_learning_rate_decays_by_cosine_to_min():
    midpoint = SCHEDULE.warmup_steps + (SCHEDULE.max_steps - SCHEDULE.warmup_steps) // 2

    assert learning_rate(midpoint, SCHEDULE) == pytest.approx((1e-3 + 1e-4) / 2)
    assert learning_rate(SCHEDULE.max_steps, SCHEDULE) == pytest.approx(1e-4)
    assert learning_rate(10_000, SCHEDULE) == pytest.approx(1e-4)


def test_train_config_rejects_warmup_longer_than_training():
    with pytest.raises(ValueError, match="warmup"):
        TrainConfig(max_steps=10, warmup_steps=10)


def test_weight_decay_skips_norm_gains():
    model = GPT(
        ModelConfig(
            vocab_size=64, context_length=8, d_model=16, n_layers=1, n_heads=2, ffn_hidden=32
        )
    )

    optimizer = build_optimizer(model, TrainConfig(), CPU)

    decayed, not_decayed = optimizer.param_groups
    assert all(p.dim() >= 2 for p in decayed["params"])
    assert all(p.dim() == 1 for p in not_decayed["params"])
    assert not_decayed["weight_decay"] == 0.0


def test_token_file_targets_are_inputs_shifted_by_one(tmp_path):
    path = tmp_path / "seq.bin"
    np.arange(100, dtype=np.uint16).tofile(path)

    inputs, targets = TokenFile(path, context_length=8).sample(
        3, torch.Generator().manual_seed(0), CPU
    )

    assert inputs.shape == targets.shape == (3, 8)
    assert torch.equal(targets, inputs + 1)


def test_verify_token_files_detects_truncated_copy(toy_setup):
    model_config, _, data_dir, tokenizer_path = toy_setup
    with (data_dir / "train.bin").open("r+b") as handle:
        handle.truncate(100)

    with pytest.raises(ValueError, match="incomplete copy"):
        verify_token_files(data_dir, tokenizer_path, model_config)


def test_training_writes_log_and_checkpoints(toy_setup, tmp_path):
    # Arrange
    model_config, train_config, data_dir, tokenizer_path = toy_setup
    ckpt_dir = tmp_path / "ckpt"

    # Act
    result = run_training(model_config, train_config, data_dir, ckpt_dir, tokenizer_path, CPU)

    # Assert
    rows = list(csv.DictReader((ckpt_dir / "log.csv").open()))
    assert [int(row["step"]) for row in rows] == [3, 6]
    assert math.isfinite(float(rows[-1]["val_bits_per_byte"]))
    assert result["step"] == 6
    assert load_checkpoint(ckpt_dir / "latest.pt")["step"] == 6
    assert (ckpt_dir / "best.pt").exists()


def test_resumed_training_matches_uninterrupted_training(toy_setup, tmp_path):
    # Arrange
    model_config, train_config, data_dir, tokenizer_path = toy_setup
    straight_dir, resumed_dir = tmp_path / "straight", tmp_path / "resumed"

    # Act: one run of 6 steps, versus 4 steps, stop, then resume to 6
    run_training(model_config, train_config, data_dir, straight_dir, tokenizer_path, CPU)
    paused = replace(train_config, stop_at_step=4)
    run_training(model_config, paused, data_dir, resumed_dir, tokenizer_path, CPU)
    run_training(model_config, train_config, data_dir, resumed_dir, tokenizer_path, CPU)

    # Assert: bit-for-bit the same weights
    straight = load_checkpoint(straight_dir / "latest.pt")["model"]
    resumed = load_checkpoint(resumed_dir / "latest.pt")["model"]
    assert straight.keys() == resumed.keys()
    assert all(torch.equal(straight[name], resumed[name]) for name in straight)


def test_resume_rejects_a_different_model_config(toy_setup, tmp_path):
    model_config, train_config, data_dir, tokenizer_path = toy_setup
    ckpt_dir = tmp_path / "ckpt"
    run_training(model_config, train_config, data_dir, ckpt_dir, tokenizer_path, CPU)
    wider = replace(model_config, d_model=64, ffn_hidden=128)

    with pytest.raises(ValueError, match="different model config"):
        run_training(wider, train_config, data_dir, ckpt_dir, tokenizer_path, CPU)
