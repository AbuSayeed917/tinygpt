"""Exact and near-duplicate removal.

Duplicated text is over-weighted during training, which drives memorisation
and leaks eval data into training (Lee et al., 2022, "Deduplicating
Training Data Makes Language Models Better").
"""

import hashlib
from collections.abc import Iterable, Iterator

import numpy as np

_MERSENNE_PRIME = np.uint64((1 << 61) - 1)
_MAX_HASH_32 = np.uint64((1 << 32) - 1)


def exact_fingerprint(text: str) -> str:
    canonical = " ".join(text.lower().split())
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def drop_exact_duplicates(texts: Iterable[str]) -> Iterator[str]:
    seen_fingerprints: set[str] = set()
    for text in texts:
        fingerprint = exact_fingerprint(text)
        if fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(fingerprint)
        yield text


def _shingles(text: str, shingle_size: int) -> set[str]:
    words = text.lower().split()
    if len(words) <= shingle_size:
        return {" ".join(words)}
    return {" ".join(words[i : i + shingle_size]) for i in range(len(words) - shingle_size + 1)}


def _hash32(shingle: str) -> int:
    return int.from_bytes(hashlib.blake2b(shingle.encode("utf-8"), digest_size=4).digest(), "big")


class MinHasher:
    """Estimates Jaccard similarity of word n-gram sets with fixed-size signatures.

    Each "permutation" is a universal hash h(x) = (a*x + b) mod p. Inputs, a and
    b all stay below 2**32, so a*x + b fits in uint64 with no overflow.
    """

    def __init__(self, num_permutations: int = 128, shingle_size: int = 5, seed: int = 0):
        if num_permutations <= 0:
            raise ValueError("num_permutations must be positive")
        if shingle_size <= 0:
            raise ValueError("shingle_size must be positive")
        rng = np.random.default_rng(seed)
        self.num_permutations = num_permutations
        self.shingle_size = shingle_size
        self._multipliers = rng.integers(1, 1 << 32, size=num_permutations, dtype=np.uint64)
        self._offsets = rng.integers(0, 1 << 32, size=num_permutations, dtype=np.uint64)

    def signature(self, text: str) -> np.ndarray:
        hashed_shingles = np.fromiter(
            (_hash32(shingle) for shingle in _shingles(text, self.shingle_size)),
            dtype=np.uint64,
        )
        # (num_shingles, 1) against (num_permutations,) -> one row per shingle.
        permuted = (hashed_shingles[:, None] * self._multipliers + self._offsets) % _MERSENNE_PRIME
        return (permuted & _MAX_HASH_32).min(axis=0)


def estimate_jaccard(signature_a: np.ndarray, signature_b: np.ndarray) -> float:
    if signature_a.shape != signature_b.shape:
        raise ValueError("Signatures must come from the same MinHasher")
    return float(np.mean(signature_a == signature_b))


def drop_near_duplicates(
    texts: Iterable[str],
    hasher: MinHasher,
    threshold: float = 0.8,
    bands: int = 16,
) -> Iterator[str]:
    """Keeps the first document of every near-duplicate cluster, streaming.

    Locality-sensitive hashing: the signature is cut into `bands`; documents
    sharing any whole band become candidates, and only candidates are compared.
    """
    if hasher.num_permutations % bands != 0:
        raise ValueError("num_permutations must be divisible by bands")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")

    rows_per_band = hasher.num_permutations // bands
    band_buckets: list[dict[bytes, list[int]]] = [{} for _ in range(bands)]
    kept_signatures: list[np.ndarray] = []

    for text in texts:
        signature = hasher.signature(text)
        band_keys = [
            signature[band * rows_per_band : (band + 1) * rows_per_band].tobytes()
            for band in range(bands)
        ]
        candidate_ids = {
            kept_id
            for band, key in enumerate(band_keys)
            for kept_id in band_buckets[band].get(key, ())
        }
        is_duplicate = any(
            estimate_jaccard(signature, kept_signatures[kept_id]) >= threshold
            for kept_id in candidate_ids
        )
        if is_duplicate:
            continue

        new_id = len(kept_signatures)
        kept_signatures.append(signature)
        for band, key in enumerate(band_keys):
            band_buckets[band].setdefault(key, []).append(new_id)
        yield text
