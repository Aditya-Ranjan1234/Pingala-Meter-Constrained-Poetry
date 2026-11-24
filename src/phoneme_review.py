"""Cross-check chandas using phoneme counts and following consonant clusters."""
import csv
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]


def pattern(slp):
    slp = re.sub(r"[^A-Za-z]", "", slp)
    nuclei = list(re.finditer("[aAiIuUfFxXeEoO]", slp))
    weights = []
    for i, match in enumerate(nuclei):
        end = nuclei[i + 1].start() if i + 1 < len(nuclei) else len(slp)
        following = slp[match.end():end]
        heavy = (match.group() in "AIUFXeEoO" or "M" in following or "H" in following
                 or len(following) >= 2 or (i == len(nuclei) - 1 and bool(following)))
        weights.append("G" if heavy else "L")
    return "".join(weights)


def main():
    sys.path.extend(str(p) for p in sorted((ROOT / ".uv-cache/archive-v0").iterdir()) if p.is_dir())
    from indic_transliteration import sanscript

    review = ROOT / "results/guided_reader_review"
    with (review / "llm_filled_reader_sheet.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    key = json.loads((review / "private_key.json").read_text(encoding="utf-8"))
    audits = []
    for row in rows:
        item = row["item"]
        slp = sanscript.transliterate(row["line"], sanscript.DEVANAGARI, sanscript.SLP1)
        observed = pattern(slp)
        expected = key[item]["scanner_pattern"]
        audits.append({"item": item, "method": key[item]["method"], "slp1": slp,
                       "phoneme_pattern": observed, "chandas_pattern": expected,
                       "exact_agreement": observed == expected})
    summary = {}
    for method in ("reference", "baseline", "guided"):
        subset = [r for r in audits if r["method"] == method]
        summary[method] = {"checked": len(subset), "agreed": sum(r["exact_agreement"] for r in subset)}
    (review / "phoneme_crosscheck.json").write_text(
        json.dumps({"summary": summary, "rows": audits}, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
