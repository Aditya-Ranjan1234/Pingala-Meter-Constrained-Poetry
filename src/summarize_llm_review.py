"""Unblind and summarize the full paired LLM review."""
import csv
import json
from pathlib import Path
from statistics import mean


def main():
    folder = Path(__file__).resolve().parents[1] / "results/guided_reader_review"
    ratings = json.loads((folder / "llm_pilot_ratings.json").read_text(encoding="utf-8"))["ratings"]
    key = json.loads((folder / "private_key.json").read_text(encoding="utf-8"))
    with (folder / "reader_sheet.csv").open(encoding="utf-8-sig", newline="") as stream:
        sheet = {row["item"]: row for row in csv.DictReader(stream)}
    if len(ratings) != 36 or {row["item"] for row in ratings} != key.keys() or sheet.keys() != key.keys():
        raise ValueError("Expected the 36 rated items to match the blinded sheet and key")
    paired = {}
    methods = {"reference", "baseline", "guided"}
    for rating in ratings:
        item = rating["item"]
        entry = key[item]
        if any(not 1 <= rating[field] <= 5 for field in ("grammar", "meaning")):
            raise ValueError(f"Invalid rating for {item}")
        paired.setdefault(entry["prompt"], {})[entry["method"]] = {**rating, "line": sheet[item]["line"]}
    if len(paired) != 12 or any(row.keys() != methods for row in paired.values()):
        raise ValueError("Expected a complete three-way comparison on 12 prompts")
    report = {"judge": "Codex LLM", "prompt_count": len(paired),
              "identical_baseline_guided_lines": sum(row["baseline"]["line"] == row["guided"]["line"]
                                                     for row in paired.values())}
    initial = json.loads((folder / "llm_scansion_spotcheck.json").read_text(encoding="utf-8"))["ratings"]
    extension = json.loads((folder / "llm_scansion_extension.json").read_text(encoding="utf-8"))
    scans = initial + extension
    if len({row["item"] for row in scans}) != len(scans):
        raise ValueError("Duplicate item in the scansion checks")
    report["exploratory_llm_scansion"] = {}
    for method in sorted(methods):
        subset = [row for row in scans if key[row["item"]]["method"] == method]
        report["exploratory_llm_scansion"][method] = {
            "checked": len(subset),
            "exact_agreement": sum(row["pattern"] == key[row["item"]]["scanner_pattern"] for row in subset),
        }
    for field in ("grammar", "meaning"):
        differences = [row["guided"][field] - row["baseline"][field] for row in paired.values()]
        report[field] = {
            "mean_by_method": {method: mean(row[method][field] for row in paired.values())
                               for method in sorted(methods)},
            "guided_minus_baseline": differences,
            "guided_higher": sum(d > 0 for d in differences),
            "tied": sum(d == 0 for d in differences),
            "baseline_higher": sum(d < 0 for d in differences),
        }
    output = json.dumps(report, indent=2)
    (folder / "llm_paired_summary.json").write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
