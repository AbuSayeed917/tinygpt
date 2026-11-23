"""Byte-level BPE: text -> UTF-8 bytes -> repeatedly merged byte pairs -> token ids.

Ids 0-255 are the raw bytes, so any string can be encoded and nothing is ever
"unknown". Merge i creates id 256 + i. Special tokens come after all merges.
"""

import json
import re
from collections.abc import Sequence
from pathlib import Path

# GPT-2's split, rewritten for the stdlib `re` module (which has no \p{L}):
# [^\W\d_] is "any letter", (?:[^\w\s]|_) is "punctuation or symbol". Splitting
# first stops merges from gluing words to the punctuation around them.
PRETOKEN_PATTERN = r"""'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+| ?\d{1,3}| ?(?:[^\w\s]|_)+|\s+(?!\S)|\s+"""
PRETOKEN_REGEX = re.compile(PRETOKEN_PATTERN)

END_OF_TEXT = "<|endoftext|>"
SYSTEM = "<|system|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
END_OF_TURN = "<|end|>"
SPECIAL_TOKENS: tuple[str, ...] = (END_OF_TEXT, SYSTEM, USER, ASSISTANT, END_OF_TURN)

FILE_FORMAT = "tinygpt-byte-bpe"
FILE_VERSION = 1
_BYTE_COUNT = 256
_CACHE_LIMIT = 200_000

Pair = tuple[int, int]


def merge_pair(symbols: Sequence[int], pair: Pair, new_id: int) -> list[int]:
    """Replaces every non-overlapping occurrence of `pair`, scanning left to right."""
    merged: list[int] = []
    index = 0
    while index < len(symbols):
        is_match = (
            index + 1 < len(symbols) and symbols[index] == pair[0] and symbols[index + 1] == pair[1]
        )
        if is_match:
            merged.append(new_id)
            index += 2
            continue
        merged.append(symbols[index])
        index += 1
    return merged


class ByteLevelBPE:
    def __init__(
        self, merges: Sequence[Pair], special_tokens: Sequence[str] = SPECIAL_TOKENS
    ) -> None:
        vocab_bytes = [bytes([byte]) for byte in range(_BYTE_COUNT)]
        for rank, (left, right) in enumerate(merges):
            known = _BYTE_COUNT + rank
            if not (0 <= left < known and 0 <= right < known):
                raise ValueError(f"Merge {rank} ({left}, {right}) refers to an unknown id")
            vocab_bytes.append(vocab_bytes[left] + vocab_bytes[right])
        if len(set(special_tokens)) != len(special_tokens):
            raise ValueError("Special tokens must be unique")

        self._merges: tuple[Pair, ...] = tuple((int(a), int(b)) for a, b in merges)
        self._merge_ranks: dict[Pair, int] = {pair: r for r, pair in enumerate(self._merges)}
        self._vocab_bytes: tuple[bytes, ...] = tuple(vocab_bytes)
        first_special_id = len(vocab_bytes)
        self._special_to_id = {tok: first_special_id + i for i, tok in enumerate(special_tokens)}
        self._id_to_special = {i: tok for tok, i in self._special_to_id.items()}
        self._special_split = re.compile(
            "(" + "|".join(re.escape(token) for token in special_tokens) + ")"
        )
        # Memo of pre-token -> ids. Web text repeats the same words constantly,
        # so this turns most of encoding into dictionary lookups.
        self._cache: dict[str, tuple[int, ...]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_bytes) + len(self._special_to_id)

    @property
    def merges(self) -> tuple[Pair, ...]:
        return self._merges

    @property
    def special_tokens(self) -> tuple[str, ...]:
        return tuple(self._special_to_id)

    def special_id(self, token: str) -> int:
        if token not in self._special_to_id:
            raise KeyError(f"Not a special token: {token!r}")
        return self._special_to_id[token]

    def _encode_pretoken(self, pretoken: str) -> tuple[int, ...]:
        cached = self._cache.get(pretoken)
        if cached is not None:
            return cached

        symbols = list(pretoken.encode("utf-8"))
        while len(symbols) > 1:
            ranked = [
                (self._merge_ranks[pair], pair)
                for pair in zip(symbols, symbols[1:], strict=False)
                if pair in self._merge_ranks
            ]
            if not ranked:
                break
            # Apply merges in the order they were learned, exactly as in training.
            rank, pair = min(ranked)
            symbols = merge_pair(symbols, pair, _BYTE_COUNT + rank)

        encoded = tuple(symbols)
        if len(self._cache) >= _CACHE_LIMIT:
            self._cache.clear()
        self._cache[pretoken] = encoded
        return encoded

    def encode(self, text: str) -> list[int]:
        """Encodes untrusted text. Never emits special ids: "<|user|>" typed by a
        user becomes ordinary characters, so roles cannot be spoofed."""
        if not isinstance(text, str):
            raise TypeError(f"encode expects str, got {type(text).__name__}")
        ids: list[int] = []
        for pretoken in PRETOKEN_REGEX.findall(text):
            ids.extend(self._encode_pretoken(pretoken))
        return ids

    def encode_with_special(self, text: str) -> list[int]:
        """Encodes trusted, template-built text in which special token strings are real."""
        ids: list[int] = []
        for piece in self._special_split.split(text):
            if piece in self._special_to_id:
                ids.append(self._special_to_id[piece])
            elif piece:
                ids.extend(self.encode(piece))
        return ids

    def token_bytes(self, token_id: int) -> bytes:
        if token_id in self._id_to_special:
            return self._id_to_special[token_id].encode("utf-8")
        if not 0 <= token_id < len(self._vocab_bytes):
            raise ValueError(f"Token id {token_id} is outside the vocabulary")
        return self._vocab_bytes[token_id]

    def decode_bytes(self, ids: Sequence[int]) -> bytes:
        return b"".join(self.token_bytes(token_id) for token_id in ids)

    def decode(self, ids: Sequence[int]) -> str:
        # A token can end halfway through a multi-byte character (e.g. an emoji
        # split across two tokens); "replace" keeps decoding total.
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    def save(self, path: Path) -> None:
        payload = {
            "format": FILE_FORMAT,
            "version": FILE_VERSION,
            "pretoken_pattern": PRETOKEN_PATTERN,
            "special_tokens": list(self.special_tokens),
            "merges": [list(pair) for pair in self._merges],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ByteLevelBPE":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != FILE_FORMAT or payload.get("version") != FILE_VERSION:
            raise ValueError(f"{path} is not a {FILE_FORMAT} v{FILE_VERSION} file")
        # Merges are only meaningful for the split they were trained with.
        if payload.get("pretoken_pattern") != PRETOKEN_PATTERN:
            raise ValueError(f"{path} was trained with a different pre-token pattern")
        merges = [(int(left), int(right)) for left, right in payload["merges"]]
        return cls(merges, tuple(payload["special_tokens"]))
