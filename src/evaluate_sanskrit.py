from __future__ import annotations
from pathlib import Path
import json
from .sanskrit import laghu_guru


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    rows = [json.loads(x) for x in (root / "data" / "sanskrit_metrical.jsonl").read_text(encoding="utf-8").splitlines()]
    evaluated = []
    for row in rows:
        observed = laghu_guru(row["text"])
        annotated = row.get("pattern", "")
        evaluated.append({"meter": row["meter"], "observed_pattern": observed, "published_pattern": annotated,
                          "syllables": len(observed)})
    summary = {"records": len(evaluated), "mean_syllables": sum(x["syllables"] for x in evaluated)/len(evaluated),
               "examples": evaluated[:20]}
    (root / "results" / "sanskrit_scansion.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
