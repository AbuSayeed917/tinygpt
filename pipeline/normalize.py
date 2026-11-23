"""Text normalisation, applied before every other filter so they all see the same text."""

import re
import unicodedata

_BLANK_LINE_RUNS = re.compile(r"\n{3,}")
_KEPT_CONTROL_CHARS = frozenset("\n\t")


def _is_printable(char: str) -> bool:
    # Unicode category "C*" covers control, format (zero-width, bidi overrides),
    # private-use and unassigned code points: invisible to readers, but they
    # waste tokens and can hide text from filters.
    return char in _KEPT_CONTROL_CHARS or not unicodedata.category(char).startswith("C")


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError(f"normalize_text expects str, got {type(text).__name__}")

    unified_newlines = text.replace("\r\n", "\n").replace("\r", "\n")
    # NFKC folds look-alike forms (full-width letters, ligatures) into one
    # spelling, so "ｆｉｌｅ" and "file" become the same tokens.
    compatible = unicodedata.normalize("NFKC", unified_newlines)
    printable = "".join(char for char in compatible if _is_printable(char))
    trimmed_lines = "\n".join(line.rstrip() for line in printable.split("\n"))
    return _BLANK_LINE_RUNS.sub("\n\n", trimmed_lines).strip()
