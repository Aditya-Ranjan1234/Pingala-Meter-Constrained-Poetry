"""Summarize the paired, independently scanned Chandomitra pilot."""
import argparse
import json
from math import comb, sqrt
from pathlib import Path


def wilson(successes, count):
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * sqrt(p * (1 - p) / count + z * z / (4 * count**2)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def pairs(report):
    return {(row["prompt"], row["external_target"]): row for row in report["rows"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("tuned", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    base, tuned = [pairs(json.loads(file.read_text(encoding="utf-8"))) for file in (args.base, args.tuned)]
    if base.keys() != tuned.keys() or not base:
        raise ValueError("Expected identical nonempty prompt and target sets")
    methods = {
        "one_draw": lambda row: row["pool"][0]["success"],
        "contrastive": lambda row: row["contrastive"]["success"],
        "full_candidate_filter_16": lambda row: any(sample["success"] for sample in row["pool"]),
        "word_boundary_filter_16": lambda row: row["word_boundary_pick"] is not None,
        "anustubh_cadence_filter_16": lambda row: row["cadence_pick"] is not None,
        "anustubh_cadence_full_filter_16": lambda row: any(sample["cadence_success"] for sample in row["pool"]),
        "anustubh_cadence_one_draw": lambda row: row["pool"][0]["cadence_success"],
        "anustubh_cadence_contrastive": lambda row: row["contrastive"]["cadence_success"],
    }
    summary = {}
    for method, successful in methods.items():
        outcomes = [(successful(base[key]), successful(tuned[key])) for key in base]
        count = len(outcomes)
        base_only = sum(before and not after for before, after in outcomes)
        tuned_only = sum(after and not before for before, after in outcomes)
        discordant = base_only + tuned_only
        p_value = min(1.0, 2 * sum(comb(discordant, k) for k in range(min(base_only, tuned_only) + 1))
                      / 2**discordant)
        summary[method] = {
            "prompts": count,
            "base_successes": sum(before for before, _ in outcomes),
            "tuned_successes": sum(after for _, after in outcomes),
            "base_wilson_95": wilson(sum(before for before, _ in outcomes), count),
            "tuned_wilson_95": wilson(sum(after for _, after in outcomes), count),
            "base_only": base_only, "tuned_only": tuned_only,
            "paired_exact_p_value_uncorrected": p_value,
        }
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
