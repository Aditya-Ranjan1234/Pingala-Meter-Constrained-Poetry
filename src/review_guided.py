"""Prepare anonymous Sanskrit review sheets and score returned annotations."""
import argparse
import csv
import json
from math import comb
from pathlib import Path
import random
import statistics
import runpy
import sys


def prepare(source, destination):
    study = json.loads(source.read_text(encoding="utf-8"))
    if len(study["rows"]) != 64 or len({row["prompt"] for row in study["rows"]}) != 64:
        raise ValueError("The paired study must contain all 64 prompts")
    if any(not 0 < row[method]["tokens"] <= study["token_cap"]
           for row in study["rows"] for method in ("baseline", "guided")):
        raise ValueError("A method exceeded the token cap or recorded no generation")
    rng = random.Random(5137)
    items = []
    paired = [row for row in study["rows"] if row["baseline"]["text"] and row["guided"]["text"]]
    selected = rng.sample(paired, min(12, len(paired)))
    for method in ("reference", "baseline", "guided"):
        for row in selected:
            text = row["reference"] if method == "reference" else row[method]["text"]
            pattern = row["reference_pattern"] if method == "reference" else row[method]["pattern"]
            items.append((row["prompt"], text, method, pattern))
    rng.shuffle(items)
    destination.mkdir(exist_ok=True)
    key = {}
    with (destination / "reader_sheet.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["item", "prompt", "line", "lg_pattern", "metre_0_or_1",
                         "grammar_1_to_5", "meaning_1_to_5", "notes"])
        for idx, (prompt, text, method, pattern) in enumerate(items, 1):
            item = f"R{idx:03}"
            writer.writerow([item, prompt, text, "", "", "", "", ""])
            key[item] = {"method": method, "scanner_pattern": pattern, "prompt": prompt}
    (destination / "private_key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    (destination / "reader_instructions.txt").write_text(
        "Review each line independently. Enter its full light/heavy sequence (L/G).\n"
        "Metre: 1 if the line conforms to basic anustubh half-verse cadence, else 0.\n"
        "Grammar: 1 unintelligible, 3 understandable with errors, 5 well-formed.\n"
        "Meaning: 1 incoherent, 3 partially coherent, 5 clear and coherent.\n"
        "Use notes to mark uncertain syllables and cross-word scansion decisions.\n"
        "Leave a score blank if you cannot assess it. Do not consult an automated scanner.\n",
        encoding="utf-8",
    )
    summary = {}
    for method in ("baseline", "guided"):
        records = [row[method] for row in study["rows"]]
        successes = [r for r in records if r["text"]]
        summary[method] = {
            "successes": len(successes), "failures": len(records) - len(successes),
            "mean_tokens_to_success": statistics.mean(r["tokens"] for r in successes) if successes else None,
            "median_tokens_to_success": statistics.median(r["tokens"] for r in successes) if successes else None,
            "total_generated_tokens": sum(r["tokens"] for r in records),
        }
    summary["guided_only"] = sum(bool(r["guided"]["text"] and not r["baseline"]["text"]) for r in study["rows"])
    summary["baseline_only"] = sum(bool(r["baseline"]["text"] and not r["guided"]["text"]) for r in study["rows"])
    both = [row for row in study["rows"] if row["baseline"]["text"] and row["guided"]["text"]]
    summary["both_succeed"] = {
        "prompts": len(both),
        **{method: statistics.median(row[method]["tokens"] for row in both) if both else None
           for method in ("baseline", "guided")},
    }
    n = summary["guided_only"] + summary["baseline_only"]
    m = min(summary["guided_only"], summary["baseline_only"])
    summary["paired_exact_p"] = min(1.0, 2 * sum(comb(n, i) for i in range(m + 1)) / 2**n)
    from .guided_study import ROOT, viable
    cache = ROOT / ".uv-cache/archive-v0"
    sys.path.extend(str(p) for p in sorted(cache.iterdir()) if p.is_dir())
    weights = runpy.run_path(str(next(cache.rglob("chandas/syllabize.py"))))["to_weight_list"]
    rejects = []
    boundaries = 0
    for row in study["rows"]:
        words = row["reference"].split()
        for end in range(1, len(words)):
            prefix = " ".join(words[:end])
            boundaries += 1
            try:
                pattern = "".join(weights(prefix))
            except ValueError:
                continue
            if pattern and not viable(pattern):
                rejects.append({"prefix": prefix, "pattern": pattern, "reference": row["reference"]})
    summary["reference_boundary_audit"] = {"boundaries": boundaries, "false_prunes": rejects}
    (destination / "automatic_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary)


def score(key_file, annotations, output):
    key = json.loads(key_file.read_text(encoding="utf-8"))
    results = []
    for path in annotations:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if {row["item"] for row in rows} != key.keys() or len(rows) != len(key):
            raise ValueError(f"Unexpected or missing item IDs in {path}")
        for method in ("reference", "baseline", "guided"):
            subset = [row for row in rows if key[row["item"]]["method"] == method]
            patterns = [(row, "".join(row["lg_pattern"].upper().split())) for row in subset if row["lg_pattern"].strip()]
            exact = sum(pattern == key[row["item"]]["scanner_pattern"] for row, pattern in patterns)
            fields = {"metre_0_or_1": (0, 1), "grammar_1_to_5": (1, 5), "meaning_1_to_5": (1, 5)}
            reports = {}
            for field, (low, high) in fields.items():
                values = [int(row[field]) for row in subset if row[field].strip()]
                if any(not low <= x <= high for x in values):
                    raise ValueError(f"Out-of-range {field} in {path}")
                reports[field] = {"n": len(values), "mean": statistics.mean(values) if values else None}
            disagreements = [{"item": row["item"], "reader": pattern,
                              "scanner": key[row["item"]]["scanner_pattern"], "notes": row["notes"]}
                             for row, pattern in patterns if pattern != key[row["item"]]["scanner_pattern"]]
            results.append({"file": str(path), "method": method, "scanner_exact": exact,
                            "annotated_patterns": len(patterns), "scores": reports,
                            "scansion_disagreements": disagreements})
    if not any(row["annotated_patterns"] or any(x["n"] for x in row["scores"].values()) for row in results):
        raise ValueError("No reader annotations were supplied")
    serialized = json.dumps(results, indent=2, ensure_ascii=False)
    if output:
        output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("prepare")
    make.add_argument("source", type=Path)
    make.add_argument("destination", type=Path)
    apply = commands.add_parser("score")
    apply.add_argument("key", type=Path)
    apply.add_argument("annotations", type=Path, nargs="+")
    apply.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.source, args.destination)
    else:
        score(args.key, args.annotations, args.output)
