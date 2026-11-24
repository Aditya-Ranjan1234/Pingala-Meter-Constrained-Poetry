"""Devanāgarī akṣara scanning and Pingala Laghu/Guru classification."""
from __future__ import annotations
import re

SHORT = set("अइउऋऌ")
LONG = set("आईऊॠएऐओऔ")
VOWELS = SHORT | LONG
MARKS = "ािीुूृॄॅॆेैॉोौंःँ॒॑"
CONSONANTS = set("कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहळ")


def _is_devanagari(ch: str) -> bool:
    return "\u0900" <= ch <= "\u097f"


def aksharas(text: str) -> list[str]:
    """Split Devanāgarī into vowel-bearing orthographic syllable units."""
    units: list[str] = []
    current = ""
    for ch in text:
        if not _is_devanagari(ch):
            if current:
                units.append(current); current = ""
            continue
        current += ch
        if ch in VOWELS or ch in "ािीुूृॄॅॆेैॉोौ":
            units.append(current); current = ""
    if current:
        if units: units[-1] += current
        else: units.append(current)
    return units


def classify_akshara(unit: str, final: bool = False) -> str:
    vowel = next((c for c in unit if c in VOWELS), "अ")
    guru = vowel in LONG or any(c in unit for c in "ंःँ")
    # A virāma or a consonant cluster closes the syllable.
    guru = guru or "्" in unit or bool(re.search(r"[कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहळ]$", unit))
    # Final short syllables may be treated as guru in traditional scansion.
    return "G" if guru else "L"


def laghu_guru(text: str) -> str:
    units = aksharas(text)
    return "".join(classify_akshara(unit, i == len(units) - 1) for i, unit in enumerate(units))


def normalize_verse(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("॥", " ").replace("।", " ")).strip()
