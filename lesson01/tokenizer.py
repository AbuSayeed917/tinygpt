"""Step 1: turn text into numbers.

A neural network can only do maths on numbers, so the very first job is a
reversible mapping text <-> list of integers ("token ids").

Lesson 1 uses the simplest possible scheme: one token per character.
Real models (GPT, Claude) use byte-level BPE, where common chunks like
" the" or "ing" become single tokens. We build that in lesson 3.
"""


class CharTokenizer:
    def __init__(self, text: str):
        if not text:
            raise ValueError("Cannot build a vocabulary from empty text")
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.char_to_id = {ch: i for i, ch in enumerate(self.chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text: str) -> list[int]:
        unknown = set(text) - set(self.char_to_id)
        if unknown:
            raise ValueError(f"Characters not in vocabulary: {sorted(unknown)}")
        return [self.char_to_id[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_char[i] for i in ids)


if __name__ == "__main__":
    text = open("../data/shakespeare.txt").read()
    tok = CharTokenizer(text)
    print(f"vocab size: {tok.vocab_size}")
    print(f"vocabulary: {''.join(tok.chars)!r}")
    sample = "Hello there"
    ids = tok.encode(sample)
    print(f"{sample!r} -> {ids} -> {tok.decode(ids)!r}")
