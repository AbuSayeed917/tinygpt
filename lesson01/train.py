"""Step 5: training = show the model text, measure how wrong its next-token
guesses are, nudge every weight to be a little less wrong. Repeat thousands of times.

Run:  python3 train.py            (about 5-10 min on an M-series Mac)
      python3 train.py --steps 300   (quick smoke test)
"""

import argparse
import math
import time
from pathlib import Path

import torch

from model import GPT, GPTConfig
from tokenizer import CharTokenizer

DATA_PATH = Path(__file__).parent.parent / "data" / "shakespeare.txt"
CHECKPOINT_PATH = Path(__file__).parent / "checkpoint.pt"


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"   # Apple Silicon GPU
    if torch.cuda.is_available():
        return "cuda"  # NVIDIA GPU
    return "cpu"


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str):
    """Pick random windows of text. y is x shifted one character to the left,
    so y[t] is the "correct answer" for position t. One window of length T
    therefore gives T training examples at once."""
    starts = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[s: s + block_size] for s in starts])
    y = torch.stack([data[s + 1: s + block_size + 1] for s in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, splits, block_size, batch_size, device, eval_batches=20):
    """Average over several batches: a single batch's loss is too noisy to trust."""
    model.eval()  # switches dropout off
    results = {}
    for name, data in splits.items():
        losses = [model(*get_batch(data, block_size, batch_size, device))[1].item()
                  for _ in range(eval_batches)]
        results[name] = sum(losses) / len(losses)
    model.train()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=250)
    args = parser.parse_args()

    torch.manual_seed(1337)  # same seed -> same run, so experiments are comparable
    device = pick_device()

    text = DATA_PATH.read_text()
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    # Hold out the last 10% the model never trains on. If train loss falls but
    # val loss rises, the model is memorising instead of learning patterns.
    split_at = int(0.9 * len(data))
    splits = {"train": data[:split_at], "val": data[split_at:]}

    config = GPTConfig(vocab_size=tokenizer.vocab_size)
    model = GPT(config).to(device)
    print(f"device: {device} | params: {model.num_params() / 1e6:.2f}M | "
          f"vocab: {tokenizer.vocab_size} | train tokens: {len(splits['train']):,}")
    print(f"loss of a random guesser = ln({tokenizer.vocab_size}) = "
          f"{math.log(tokenizer.vocab_size):.2f}  <- our starting point\n")

    # AdamW: gradient descent that adapts the step size per weight and gently
    # shrinks weights toward zero (weight decay). The default for LLMs.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    start = time.time()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0:
            losses = estimate_loss(model, splits, config.block_size, args.batch_size, device)
            print(f"step {step:5d} | train {losses['train']:.3f} | val {losses['val']:.3f} "
                  f"| {time.time() - start:5.0f}s")

        x, y = get_batch(splits["train"], config.block_size, args.batch_size, device)
        _, loss = model(x, y)              # forward pass: make predictions, score them
        optimizer.zero_grad(set_to_none=True)
        loss.backward()                    # backprop: how should each weight change?
        optimizer.step()                   # apply the change

    torch.save({"model": model.state_dict(), "config": config, "chars": tokenizer.chars},
               CHECKPOINT_PATH)
    print(f"\nsaved {CHECKPOINT_PATH.name}")


if __name__ == "__main__":
    main()
