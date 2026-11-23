import pytest

from pipeline.dedup import (
    MinHasher,
    drop_exact_duplicates,
    drop_near_duplicates,
    estimate_jaccard,
    exact_fingerprint,
)


def test_exact_fingerprint_ignores_case_and_whitespace():
    assert exact_fingerprint("Hello   World\n") == exact_fingerprint("hello world")


def test_drop_exact_duplicates_keeps_first_occurrence_in_order():
    texts = ["a b", "c d", "A  B", "e f"]

    result = list(drop_exact_duplicates(texts))

    assert result == ["a b", "c d", "e f"]


def test_minhash_signature_is_deterministic_for_a_seed(good_paragraph):
    first = MinHasher(seed=7).signature(good_paragraph)
    second = MinHasher(seed=7).signature(good_paragraph)

    assert (first == second).all()
    assert first.shape == (128,)


def test_minhash_estimates_high_similarity_for_one_word_edit(good_paragraph):
    hasher = MinHasher()
    edited = good_paragraph.replace("market", "fair")

    similarity = estimate_jaccard(hasher.signature(good_paragraph), hasher.signature(edited))

    # True Jaccard of the 5-gram sets here is about 0.86.
    assert similarity > 0.7


def test_minhash_estimates_low_similarity_for_unrelated_text(good_paragraph):
    hasher = MinHasher()
    other = "Quantum computers use qubits that can hold superpositions of states " * 3

    similarity = estimate_jaccard(hasher.signature(good_paragraph), hasher.signature(other))

    assert similarity < 0.1


def test_drop_near_duplicates_removes_lightly_edited_copy(good_paragraph):
    texts = [
        good_paragraph,
        good_paragraph + " The end.",
        "A completely different document about compilers, parsers and type systems.",
    ]

    result = list(drop_near_duplicates(texts, MinHasher(), threshold=0.8))

    assert result == [texts[0], texts[2]]


def test_drop_near_duplicates_rejects_bands_that_do_not_divide_permutations():
    with pytest.raises(ValueError):
        list(drop_near_duplicates(["x"], MinHasher(num_permutations=100), bands=16))
