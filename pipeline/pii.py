"""Redact personal data so the model cannot memorise and later repeat it."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b")
_IPV4_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
_IPV4 = re.compile(rf"\b{_IPV4_OCTET}(?:\.{_IPV4_OCTET}){{3}}\b")
# Needs at least 8 digits in groups, so years ("2019-2020") and short numbers survive.
_PHONE = re.compile(
    r"(?<![\w.])(?:\+\d{1,3}[\s.-]?)?\(?\d{2,5}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,4}(?![\w])"
)

# Order matters: an IPv4 address also looks like a dotted phone number.
_RULES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("email", _EMAIL, "[EMAIL]"),
    ("ipv4", _IPV4, "[IP]"),
    ("phone", _PHONE, "[PHONE]"),
)


@dataclass(frozen=True)
class RedactionResult:
    text: str
    counts: Mapping[str, int]

    @property
    def has_redactions(self) -> bool:
        return any(self.counts.values())


def redact_pii(text: str) -> RedactionResult:
    if not isinstance(text, str):
        raise TypeError(f"redact_pii expects str, got {type(text).__name__}")

    redacted = text
    counts: dict[str, int] = {}
    for name, pattern, placeholder in _RULES:
        redacted, counts[name] = pattern.subn(placeholder, redacted)
    return RedactionResult(text=redacted, counts=MappingProxyType(counts))
