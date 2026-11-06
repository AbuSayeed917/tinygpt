"""Step 6: generate text from a trained checkpoint.

python3 sample.py
python3 sample.py --prompt "ROMEO:" --temperature 0.6 --tokens 300
"""

import argparse
from pathlib import Path

import torch

from model import GPT
from tokenizer import CharTokenizer
from train import pick_device

CHECKPOINT_PATH = Path(__file__).parent / "checkpoint.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="\n")
    parser.add_argument("--tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError("No checkpoint.pt yet - run `python3 train.py` first")

    device = pick_device()
    # weights_only=False because the checkpoint also stores our GPTConfig object.
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    tokenizer = CharTokenizer("".join(checkpoint["chars"]))
    model = GPT(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    idx = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    out = model.generate(idx, args.tokens, temperature=args.temperature, top_k=args.top_k)
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
