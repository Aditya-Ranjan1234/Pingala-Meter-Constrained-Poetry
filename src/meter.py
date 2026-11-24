"""Meter utilities based on Pingala's binary prosodic enumeration."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import re

LAGHU = "L"
GURU = "G"


def prastara(length: int) -> list[str]:
    """Return all binary patterns of *length* in lexicographic Prastāra order."""
    if length < 0:
        raise ValueError("length must be non-negative")
    return ["".join(bits) for bits in product((LAGHU, GURU), repeat=length)]


def pattern_to_matra(pattern: str) -> int:
    """Map Laghu/Guru symbols to mātrā count (1/2)."""
    if any(ch not in (LAGHU, GURU) for ch in pattern):
        raise ValueError("pattern must contain only L and G")
    return sum(1 if ch == LAGHU else 2 for ch in pattern)


def valid_patterns(syllables: int, target_matra: int | None = None) -> list[str]:
    patterns = prastara(syllables)
    return [p for p in patterns if target_matra is None or pattern_to_matra(p) == target_matra]


VOWELS = "aeiouāīūṛṝḷeaiou"
LONG_VOWELS = set("āīūṝeō")


def syllabify(text: str) -> list[str]:
    """A conservative Latin/Sanskrit transliteration syllabifier for the prototype."""
    words = re.findall(r"[A-Za-zāīūṛṝḷṅñṭḍṇśṣṃḥ]+", text.lower())
    syllables: list[str] = []
    for word in words:
        current = ""
        for char in word:
            current += char
            if char in VOWELS:
                syllables.append(current)
                current = ""
        if current and syllables:
            syllables[-1] += current
    return syllables


def classify_syllables(text: str) -> str:
    """Classify syllables as L/G using vowel length and closed-syllable weight."""
    result = []
    for syllable in syllabify(text):
        vowel = next((c for c in syllable if c in VOWELS), "a")
        closed = bool(re.search(r"[^aeiouāīūṛṝḷṅñṭḍṇśṣṃḥ]$", syllable))
        result.append(GURU if vowel in LONG_VOWELS or closed else LAGHU)
    return "".join(result)


@dataclass(frozen=True)
class Meter:
    pattern: str

    def accepts(self, candidate: str) -> bool:
        return candidate == self.pattern

    @property
    def syllables(self) -> int:
        return len(self.pattern)
