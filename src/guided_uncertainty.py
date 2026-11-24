"""Paired bootstrap over the complete prompt set, including failed attempts."""
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = json.loads((ROOT / "results/guided_fresh_64.json").read_text(encoding="utf-8"))
    differences = [(int(bool(r["guided"]["text"])) - int(bool(r["baseline"]["text"])),
                    r["baseline"]["tokens"] - r["guided"]["tokens"]) for r in data["rows"]]
    n = len(differences)
    if n != 64:
        raise ValueError("Expected 64 completed pairs")
    draws = 20000
    rng = random.Random(922)
    simulated = [tuple(sum(column) / n for column in zip(*rng.choices(differences, k=n)))
                 for _ in range(draws)]
    report = {"prompts": n, "resamples": draws, "seed": 922}
    report["selected_outputs_with_exact_reference_pattern"] = {
        method: sum(bool(row[method]["text"]) and row[method]["pattern"] == row["reference_pattern"]
                    for row in data["rows"])
        for method in ("baseline", "guided")
    }
    report["coverage_by_observed_token_threshold"] = {
        cap: {method: sum(bool(row[method]["text"]) and row[method]["tokens"] <= cap
                          for row in data["rows"])
              for method in ("baseline", "guided")}
        for cap in (512, 1024, 1536, 2048, 3072, data["token_cap"])
    }
    for index, name in enumerate(("coverage_difference", "sampled_tokens_saved_per_prompt")):
        values = sorted(row[index] for row in simulated)
        report[name] = {"estimate": sum(row[index] for row in differences) / n,
                        "paired_percentile_95": [values[int(.025 * draws)], values[int(.975 * draws) - 1]]}
    (ROOT / "results/guided_reader_review/paired_uncertainty.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
