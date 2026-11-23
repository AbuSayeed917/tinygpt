"""Pretrain the model on the token files, resumably.

    python -m train.pretrain --data-dir data/tokens --ckpt-dir checkpoints/pretrain

If <ckpt-dir>/latest.pt exists, training resumes from it exactly: model,
optimiser, loss scaler, step and the batch sampler's random state are all
restored. Colab can disconnect at any time, so this is the core feature.
"""

import argparse
import contextlib
import csv
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch

from bpe.encode_corpus import sha256_of_file
from bpe.tokenizer import END_OF_TEXT, ByteLevelBPE
from model.gpt import GPT, ModelConfig

CHECKPOINT_FORMAT = "tinygpt-pretrain-checkpoint"
CHECKPOINT_VERSION = 1
_LOG_FIELDS = (
    "step",
    "tokens_seen",
    "train_loss",
    "val_loss",
    "val_bits_per_byte",
    "learning_rate",
    "grad_norm",
    "tokens_per_second",
    "elapsed_minutes",
)
_SAMPLE_PROMPT = "The"


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 32
    grad_accum_steps: int = 4
    max_steps: int = 5_400
    peak_lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_batches: int = 40
    checkpoint_every_minutes: float = 10.0
    max_minutes: float | None = None
    stop_at_step: int | None = None
    seed: int = 1337
    compile: bool = False

    def __post_init__(self) -> None:
        if self.warmup_steps >= self.max_steps:
            raise ValueError("warmup_steps must be smaller than max_steps")
        if not 0 < self.min_lr <= self.peak_lr:
            raise ValueError("need 0 < min_lr <= peak_lr")
        for name in ("batch_size", "grad_accum_steps", "eval_every", "eval_batches"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


def learning_rate(step: int, config: TrainConfig) -> float:
    """Linear warmup, then cosine decay from peak_lr to min_lr at max_steps."""
    if step < config.warmup_steps:
        return config.peak_lr * (step + 1) / config.warmup_steps
    if step >= config.max_steps:
        return config.min_lr
    progress = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return config.min_lr + cosine * (config.peak_lr - config.min_lr)


def build_optimizer(model: GPT, config: TrainConfig, device: torch.device) -> torch.optim.AdamW:
    # Weight decay pulls weights toward zero; that helps matrices generalise but
    # would fight RMSNorm gains, whose job is to scale, so 1-D params are exempt.
    parameters = list(model.named_parameters())
    groups = [
        {"params": [p for _, p in parameters if p.dim() >= 2], "weight_decay": config.weight_decay},
        {"params": [p for _, p in parameters if p.dim() < 2], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=config.peak_lr,
        betas=(config.beta1, config.beta2),
        fused=device.type == "cuda",
    )


class TokenFile:
    """Random fixed-length windows from a uint16 token file, read via memmap."""

    def __init__(self, path: Path, context_length: int) -> None:
        self.tokens = np.memmap(path, dtype="<u2", mode="r")
        self.context_length = context_length
        if len(self.tokens) <= context_length + 1:
            raise ValueError(f"{path} has too few tokens for context {context_length}")

    def sample(
        self, batch_size: int, generator: torch.Generator, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        highest_start = len(self.tokens) - self.context_length - 1
        starts = torch.randint(highest_start, (batch_size,), generator=generator).tolist()
        windows = np.stack([self.tokens[s : s + self.context_length + 1] for s in starts])
        batch = torch.from_numpy(windows.astype(np.int64))
        inputs, targets = batch[:, :-1], batch[:, 1:]
        if device.type == "cuda":
            return (
                inputs.pin_memory().to(device, non_blocking=True),
                targets.pin_memory().to(device, non_blocking=True),
            )
        return inputs.to(device), targets.to(device)


@dataclass(frozen=True)
class Precision:
    autocast_dtype: torch.dtype | None
    uses_loss_scaling: bool


def choose_precision(device: torch.device) -> Precision:
    if device.type != "cuda":
        return Precision(autocast_dtype=None, uses_loss_scaling=False)
    # bf16 needs Ampere (8.0+). The T4 is 7.5, so it gets fp16 plus loss scaling,
    # which stops small gradients rounding to zero in fp16.
    if torch.cuda.get_device_capability(device) >= (8, 0):
        return Precision(autocast_dtype=torch.bfloat16, uses_loss_scaling=False)
    return Precision(autocast_dtype=torch.float16, uses_loss_scaling=True)


def _autocast(device: torch.device, precision: Precision) -> contextlib.AbstractContextManager:
    if precision.autocast_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=precision.autocast_dtype)


def verify_token_files(data_dir: Path, tokenizer_path: Path, model_config: ModelConfig) -> dict:
    meta = json.loads((data_dir / "meta.json").read_text())
    if meta["vocab_size"] != model_config.vocab_size:
        raise ValueError(
            f"Token files use vocab {meta['vocab_size']}, model expects {model_config.vocab_size}"
        )
    if meta["tokenizer_sha256"] != sha256_of_file(tokenizer_path):
        raise ValueError("Token files were made with a different tokenizer file")
    for split in ("train", "val"):
        expected_bytes = meta["tokens"][split] * 2
        actual_bytes = (data_dir / f"{split}.bin").stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"{split}.bin is {actual_bytes} bytes, expected {expected_bytes} (incomplete copy?)"
            )
    return meta


def bytes_per_token(val_path: Path, tokenizer: ByteLevelBPE) -> float:
    """UTF-8 bytes per token on validation, to convert loss into bits per byte."""
    tokens = np.memmap(val_path, dtype="<u2", mode="r")
    end_of_text = tokenizer.special_id(END_OF_TEXT)
    text_ids = tokens[tokens != end_of_text].tolist()
    return len(tokenizer.decode_bytes(text_ids)) / len(tokens)


@torch.no_grad()
def evaluate(
    model: GPT,
    val: TokenFile,
    config: TrainConfig,
    device: torch.device,
    precision: Precision,
) -> float:
    # A fixed seed means every evaluation scores the same batches, so changes in
    # validation loss come from the model, not from sampling noise.
    generator = torch.Generator().manual_seed(config.seed + 1)
    model.eval()
    losses = []
    for _ in range(config.eval_batches):
        inputs, targets = val.sample(config.batch_size, generator, device)
        with _autocast(device, precision):
            _, loss = model(inputs, targets)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError) as error:
        print(f"Could not read git commit ({error}); recording 'unknown'")
        return "unknown"
    return result.stdout.strip()


def save_checkpoint(path: Path, state: dict) -> None:
    # Save then rename: a disconnect mid-write leaves the previous checkpoint intact.
    partial = path.with_name(path.name + ".partial")
    torch.save(state, partial)
    os.replace(partial, path)


def load_checkpoint(path: Path) -> dict:
    # weights_only=True refuses arbitrary pickled objects, so loading a
    # checkpoint can never execute code.
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") != CHECKPOINT_FORMAT or state.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} v{CHECKPOINT_VERSION} file")
    return state


def _append_log(log_path: Path, row: dict[str, object]) -> None:
    is_new = not log_path.exists()
    with log_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_LOG_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def _sample_text(model: GPT, tokenizer: ByteLevelBPE, device: torch.device) -> str:
    prompt = torch.tensor([tokenizer.encode(_SAMPLE_PROMPT)], device=device)
    torch.manual_seed(0)
    room_left = model.config.context_length - prompt.size(1)
    output = model.generate(prompt, max_new_tokens=min(40, room_left), temperature=0.8, top_k=50)
    return tokenizer.decode(output[0].tolist())


def _training_steps(start: int, config: TrainConfig) -> Iterator[int]:
    stop = (
        config.max_steps
        if config.stop_at_step is None
        else min(config.stop_at_step, config.max_steps)
    )
    yield from range(start, stop)


def run_training(
    model_config: ModelConfig,
    config: TrainConfig,
    data_dir: Path,
    ckpt_dir: Path,
    tokenizer_path: Path,
    device: torch.device,
) -> dict[str, float]:
    verify_token_files(data_dir, tokenizer_path, model_config)
    tokenizer = ByteLevelBPE.load(tokenizer_path)
    train_data = TokenFile(data_dir / "train.bin", model_config.context_length)
    val_data = TokenFile(data_dir / "val.bin", model_config.context_length)
    val_bytes_per_token = bytes_per_token(data_dir / "val.bin", tokenizer)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_path, best_path, log_path = (
        ckpt_dir / "latest.pt",
        ckpt_dir / "best.pt",
        ckpt_dir / "log.csv",
    )

    torch.manual_seed(config.seed)
    model = GPT(model_config).to(device)
    optimizer = build_optimizer(model, config, device)
    precision = choose_precision(device)
    scaler = torch.amp.GradScaler("cuda", enabled=precision.uses_loss_scaling)
    sampler = torch.Generator().manual_seed(config.seed)
    start_step, tokens_seen, best_val_loss = 0, 0, float("inf")

    if latest_path.exists():
        state = load_checkpoint(latest_path)
        if state["model_config"] != asdict(model_config):
            raise ValueError("latest.pt was trained with a different model config")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        sampler.set_state(state["sampler_state"])
        start_step, tokens_seen = state["step"], state["tokens_seen"]
        best_val_loss = state["best_val_loss"]
        print(f"Resumed from step {start_step:,} ({tokens_seen:,} tokens seen)")

    trainable = torch.compile(model) if config.compile else model
    tokens_per_step = config.batch_size * config.grad_accum_steps * model_config.context_length
    commit = _git_commit()
    print(
        f"{model.num_params():,} params | {tokens_per_step:,} tokens/step | "
        f"device {device} | precision {precision.autocast_dtype or torch.float32}"
    )

    def checkpoint_state(step: int) -> dict:
        return {
            "format": CHECKPOINT_FORMAT,
            "version": CHECKPOINT_VERSION,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "sampler_state": sampler.get_state(),
            "step": step,
            "tokens_seen": tokens_seen,
            "best_val_loss": best_val_loss,
            "model_config": asdict(model_config),
            "train_config": asdict(config),
            "git_commit": commit,
        }

    started = time.monotonic()
    last_checkpoint = started
    interval_started, interval_tokens = started, 0
    completed = start_step  # steps fully applied to the weights
    last_train_loss = float("nan")
    try:
        for step in _training_steps(start_step, config):
            current_lr = learning_rate(step, config)
            for group in optimizer.param_groups:
                group["lr"] = current_lr

            accumulated_loss = 0.0
            for _ in range(config.grad_accum_steps):
                inputs, targets = train_data.sample(config.batch_size, sampler, device)
                with _autocast(device, precision):
                    _, loss = trainable(inputs, targets)
                scaler.scale(loss / config.grad_accum_steps).backward()
                accumulated_loss += loss.item() / config.grad_accum_steps
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip).item()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            tokens_seen += tokens_per_step
            interval_tokens += tokens_per_step
            last_train_loss = accumulated_loss
            completed = step + 1

            is_eval_step = completed % config.eval_every == 0 or completed == config.max_steps
            if is_eval_step:
                val_loss = evaluate(model, val_data, config, device, precision)
                now = time.monotonic()
                row = {
                    "step": completed,
                    "tokens_seen": tokens_seen,
                    "train_loss": round(accumulated_loss, 4),
                    "val_loss": round(val_loss, 4),
                    "val_bits_per_byte": round(val_loss / math.log(2) / val_bytes_per_token, 4),
                    "learning_rate": f"{current_lr:.2e}",
                    "grad_norm": round(grad_norm, 3),
                    "tokens_per_second": round(interval_tokens / (now - interval_started)),
                    "elapsed_minutes": round((now - started) / 60, 1),
                }
                _append_log(log_path, row)
                print(" | ".join(f"{key} {value}" for key, value in row.items()), flush=True)
                print(f"  sample: {_sample_text(model, tokenizer, device)!r}", flush=True)
                interval_started, interval_tokens = time.monotonic(), 0
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(best_path, checkpoint_state(completed))

            minutes_elapsed = (time.monotonic() - started) / 60
            is_checkpoint_due = (
                time.monotonic() - last_checkpoint
            ) / 60 >= config.checkpoint_every_minutes
            if is_checkpoint_due:
                save_checkpoint(latest_path, checkpoint_state(completed))
                last_checkpoint = time.monotonic()
            if config.max_minutes is not None and minutes_elapsed >= config.max_minutes:
                print(f"Time budget of {config.max_minutes} min reached at step {completed:,}")
                break
    except KeyboardInterrupt:
        # Colab's stop button raises KeyboardInterrupt. `completed` only advances
        # after an optimiser step, so a half-done step is simply redone on resume.
        optimizer.zero_grad(set_to_none=True)
        print(f"Interrupted; saving at step {completed:,}")

    save_checkpoint(latest_path, checkpoint_state(completed))
    print(f"Saved {latest_path} at step {completed:,}")
    return {"step": float(completed), "train_loss": last_train_loss, "best_val_loss": best_val_loss}


def _pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("bpe/tinygpt-bpe-8k.json"))
    parser.add_argument("--device", default="auto")
    optional_types = {"max_minutes": float, "stop_at_step": int}
    for field_name, default in asdict(TrainConfig()).items():
        flag = "--" + field_name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(flag, action="store_true")
            continue
        value_type = optional_types.get(field_name, type(default))
        parser.add_argument(flag, type=value_type, default=default)
    args = parser.parse_args(argv)

    overrides = {name: getattr(args, name) for name in asdict(TrainConfig())}
    config = replace(TrainConfig(), **overrides)
    run_training(
        ModelConfig(),
        config,
        args.data_dir,
        args.ckpt_dir,
        args.tokenizer,
        _pick_device(args.device),
    )


if __name__ == "__main__":
    main(sys.argv[1:])
