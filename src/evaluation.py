from __future__ import annotations
import json
from pathlib import Path
from .decoder import Candidate
from .meter import Meter


def evaluate(poem: list[Candidate], meter: Meter) -> dict[str, float]:
    valid = sum(meter.accepts(line.pattern) for line in poem)
    return {
        "lines": len(poem),
        "meter_validity": valid / len(poem) if poem else 0.0,
        "mean_likelihood_proxy": sum(line.score for line in poem) / len(poem) if poem else 0.0,
    }


def save_json(path: str | Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
