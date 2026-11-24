"""Repair the annotated corpus and build a larger Sanskrit fine-tuning file."""
from __future__ import annotations
from pathlib import Path
import json, re, os


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    raw = json.loads((root / "data" / "sanskrit_metrical" / "dataset.json").read_text(encoding="utf-8"))["data"]
    records = [r for r in raw if r.get("text") and r.get("meter")]
    out = root / "data" / "sanskrit_metrical.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8")
    lines = []
    for r in records:
        lines.append(f"<meter>{r['meter']}</meter> <topic>{r.get('topic','')}</topic> {r['text']}")
    # The labeled corpus is the reproducible base. Optionally append a larger
    # CC0 Sansar stream when --scale is passed and network access is available.
    if "--scale" in os.sys.argv:
        from datasets import load_dataset
        ds = load_dataset("MuseMesh/sansar-sanskrit-corpus", "cc0-1.0", split="train", streaming=True)
        for row in ds:
            text = row.get("text", "")
            if re.search(r"[\u0900-\u097f]", text):
                lines.append(text.strip())
            if len(lines) >= 50550: break
    train = root / "data" / "sanskrit_train.txt"
    train.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"annotated_records": len(records), "training_lines": len(lines), "jsonl": str(out), "train": str(train)}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
