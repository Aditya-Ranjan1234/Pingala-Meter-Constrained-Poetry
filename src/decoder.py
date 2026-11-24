"""Transparent phrase-level decoders used as an LLM-constrained decoding surrogate."""
from __future__ import annotations
from dataclasses import dataclass
from .meter import classify_syllables, Meter


@dataclass(frozen=True)
class Candidate:
    text: str
    score: float
    pattern: str


LEXICON = [
    # The explicit patterns model cached token-to-syllable annotations from an
    # LLM vocabulary; the classifier remains available for raw text inputs.
    ("a bright moon rises", 0.98, "LGGL"), ("quiet stars remember", 0.94, "GLLG"),
    ("new winds carry seed", 0.91, "GGLG"), ("the patient moon listens", 0.90, "LGGL"),
    ("bright birds cross home", 0.88, "GGGL"), ("kind hands shape tomorrow", 0.87, "LGGL"),
    ("inward fires awaken", 0.86, "GLLG"), ("green leaves greet rain", 0.84, "GGGL"),
    ("we learn from silence", 0.82, "LLGG"), ("old stones hold stories", 0.80, "GGGG"),
    ("hope returns with morning", 0.78, "LGGG"), ("clear water finds the sea", 0.76, "GGLG"),
]


def candidates() -> list[Candidate]:
    return [Candidate(text, score, pattern) for text, score, pattern in LEXICON]


def unconstrained_decode(width: int = 4, lines: int = 4) -> list[Candidate]:
    pool = sorted(candidates(), key=lambda c: c.score, reverse=True)
    return [pool[i % len(pool)] for i in range(lines)]


def constrained_decode(meter: Meter, width: int = 4, lines: int = 4) -> list[Candidate]:
    """Beam-like decoding: filter candidate expansions by the finite-state meter.

    In an LLM integration, this filter is applied incrementally to next-token
    logits; here each phrase is an atomic candidate to keep the experiment
    deterministic and dependency-free.
    """
    pool = [c for c in candidates() if meter.accepts(c.pattern)]
    if not pool:
        raise ValueError(f"no candidate realizes meter {meter.pattern}")
    pool = sorted(pool, key=lambda c: c.score, reverse=True)[:width]
    return [pool[i % len(pool)] for i in range(lines)]
