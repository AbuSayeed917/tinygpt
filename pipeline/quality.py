"""Heuristic quality filter, after the rules in DeepMind's Gopher paper (Rae et al., 2021).

Each rule rejects a typical kind of web junk: menus, SEO keyword lists,
boilerplate, machine-generated tables. They are cheap and need no model.
"""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

_STOP_WORDS = frozenset({"the", "be", "to", "of", "and", "that", "have", "with"})
_BULLETS = ("•", "-", "*", "·")
_ELLIPSES = ("...", "…")


@dataclass(frozen=True)
class QualityThresholds:
    min_words: int = 50
    max_words: int = 100_000
    min_mean_word_length: float = 3.0
    max_mean_word_length: float = 10.0
    max_symbol_to_word_ratio: float = 0.1
    max_bullet_line_fraction: float = 0.9
    max_ellipsis_line_fraction: float = 0.3
    min_alphabetic_word_fraction: float = 0.8
    min_stop_words: int = 2
    max_duplicate_line_fraction: float = 0.3
    max_top_bigram_char_fraction: float = 0.2


@dataclass(frozen=True)
class QualityVerdict:
    failed_rules: tuple[str, ...]
    stats: Mapping[str, float]

    @property
    def is_kept(self) -> bool:
        return not self.failed_rules


def _top_bigram_char_fraction(words: list[str]) -> float:
    if len(words) < 2:
        return 0.0
    bigram_counts = Counter(zip(words, words[1:], strict=False))
    (top_first, top_second), occurrences = bigram_counts.most_common(1)[0]
    total_chars = sum(len(word) for word in words)
    return occurrences * (len(top_first) + len(top_second)) / total_chars


def compute_quality_stats(text: str) -> dict[str, float]:
    words = text.split()
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    word_count = len(words)
    line_count = len(lines)
    if word_count == 0 or line_count == 0:
        return {"word_count": 0.0}

    symbol_count = text.count("#") + sum(text.count(ellipsis) for ellipsis in _ELLIPSES)
    lowercase_words = [word.lower() for word in words]
    return {
        "word_count": float(word_count),
        "mean_word_length": sum(len(word) for word in words) / word_count,
        "symbol_to_word_ratio": symbol_count / word_count,
        "bullet_line_fraction": sum(line.startswith(_BULLETS) for line in lines) / line_count,
        "ellipsis_line_fraction": sum(line.endswith(_ELLIPSES) for line in lines) / line_count,
        "alphabetic_word_fraction": sum(any(char.isalpha() for char in word) for word in words)
        / word_count,
        "stop_word_count": float(sum(word in _STOP_WORDS for word in lowercase_words)),
        "duplicate_line_fraction": 1 - len(set(lines)) / line_count,
        "top_bigram_char_fraction": _top_bigram_char_fraction(lowercase_words),
    }


def _failed_rules(stats: Mapping[str, float], limits: QualityThresholds) -> tuple[str, ...]:
    checks = {
        "too_few_words": stats["word_count"] < limits.min_words,
        "too_many_words": stats["word_count"] > limits.max_words,
        "mean_word_length": not (
            limits.min_mean_word_length <= stats["mean_word_length"] <= limits.max_mean_word_length
        ),
        "symbol_to_word_ratio": stats["symbol_to_word_ratio"] > limits.max_symbol_to_word_ratio,
        "bullet_lines": stats["bullet_line_fraction"] > limits.max_bullet_line_fraction,
        "ellipsis_lines": stats["ellipsis_line_fraction"] > limits.max_ellipsis_line_fraction,
        "non_alphabetic_words": (
            stats["alphabetic_word_fraction"] < limits.min_alphabetic_word_fraction
        ),
        "too_few_stop_words": stats["stop_word_count"] < limits.min_stop_words,
        "duplicate_lines": stats["duplicate_line_fraction"] > limits.max_duplicate_line_fraction,
        "repeated_bigram": (
            stats["top_bigram_char_fraction"] > limits.max_top_bigram_char_fraction
        ),
    }
    return tuple(name for name, is_failed in checks.items() if is_failed)


def assess_quality(text: str, limits: QualityThresholds | None = None) -> QualityVerdict:
    if not isinstance(text, str):
        raise TypeError(f"assess_quality expects str, got {type(text).__name__}")

    active_limits = limits or QualityThresholds()
    stats = compute_quality_stats(text)
    if stats["word_count"] == 0:
        return QualityVerdict(failed_rules=("empty",), stats=MappingProxyType(stats))
    return QualityVerdict(
        failed_rules=_failed_rules(stats, active_limits),
        stats=MappingProxyType(stats),
    )
