"""Check a saved generation pool with the external chandas syllable weights."""
import argparse
import json
import logging
from pathlib import Path
import re
import runpy
import sys

from .sanskrit import laghu_guru


def devanagari_key(text):
    return re.sub(r"[^\u0900-\u0963]", "", text)


def has_anustubh_cadence(pattern):
    """Apply the basic half-verse cadence in Chandomitra, section 2.2."""
    return bool(pattern and len(pattern) == 16 and pattern[4:7] == "LGG"
                and pattern[12:15] == "LGL")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("generation", type=Path)
    parser.add_argument("splits", type=Path)
    parser.add_argument("scanner_source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    if args.cache:
        sys.path.extend(str(path) for path in sorted(args.cache.iterdir()) if path.is_dir())
    weights = runpy.run_path(str(args.scanner_source))["to_weight_list"]
    report = json.loads(args.generation.read_text(encoding="utf-8"))
    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    corpus = {devanagari_key(line) for lines in splits.values() for line in lines}
    logging.disable(logging.CRITICAL)
    refs = {}
    for text in splits["test"]:
        words = re.findall(r"[\u0900-\u0963]+", text)
        if len(words) > 1 and len(laghu_guru(text)) >= 3:
            refs.setdefault(words[0], text)

    def scan(text):
        if "\ufffd" in text:
            return None
        try:
            return "".join(weights(text)) or None
        except ValueError:
            return None

    output = []
    for row in report["rows"]:
        reference = refs[row["prompt"]]
        target = scan(reference)

        def complete_prefix(candidate, accepts):
            words = candidate["text"].split()
            for end in range(2, len(words)):
                text = " ".join(words[:end])
                if not text.startswith(row["prompt"] + " "):
                    continue
                if not re.fullmatch(r"[\u0900-\u0963 ]+", text):
                    continue
                if devanagari_key(text) in corpus:
                    continue
                if accepts(scan(text)):
                    return text
            return None

        def audit(candidate):
            pattern = scan(candidate["text"])
            eligible = bool(candidate["added_text"]
                            and devanagari_key(candidate["text"]) not in corpus
                            and re.fullmatch(r"[\u0900-\u0963 ]+", candidate["text"]))
            return {"text": candidate["text"], "external_pattern": pattern,
                    "eligible": eligible,
                    "success": bool(eligible and target and target == pattern),
                    "cadence_success": eligible and has_anustubh_cadence(pattern)}

        pool = [audit(candidate) for candidate in row["pool"]]
        selected = row["selected_index"]
        prefix_pick = next((text for candidate in row["pool"]
                            if (text := complete_prefix(candidate, lambda p: bool(target and p == target)))
                            is not None), None)
        cadence_prefixes = [complete_prefix(candidate, has_anustubh_cadence) for candidate in row["pool"]]
        cadence_pick = next((text for text in cadence_prefixes if text is not None), None)
        first_cadence = next((i for i, (candidate, prefix) in enumerate(zip(pool, cadence_prefixes), 1)
                              if candidate["cadence_success"] or prefix is not None), None)
        cadence_selection = None
        if first_cadence is not None:
            i = first_cadence - 1
            text = pool[i]["text"] if pool[i]["cadence_success"] else cadence_prefixes[i]
            cadence_selection = {"text": text, "pattern": scan(text),
                                 "source": "full" if pool[i]["cadence_success"] else "word_boundary"}
        output.append({"prompt": row["prompt"], "reference": reference, "external_target": target,
                       "internal_target": row["target"], "pool": pool,
                       "contrastive": audit(row["contrastive"]),
                       "internal_pick": None if selected is None else pool[selected],
                       "word_boundary_pick": prefix_pick, "cadence_pick": cadence_pick,
                       "first_cadence_draw": first_cadence, "cadence_selection": cadence_selection})
    summary = {
        "prompts": len(output),
        "target_agreement": sum(row["external_target"] == row["internal_target"] for row in output),
        "external_reference_scan_failures": sum(row["external_target"] is None for row in output),
        "one_draw_success": sum(row["pool"][0]["success"] for row in output),
        "contrastive_success": sum(row["contrastive"]["success"] for row in output),
        "internal_picks_confirmed": sum(bool(row["internal_pick"] and row["internal_pick"]["success"]) for row in output),
        "external_filter_prompt_success": sum(any(x["success"] for x in row["pool"]) for row in output),
        "word_boundary_filter_prompt_success": sum(row["word_boundary_pick"] is not None for row in output),
        "anustubh_reference_cadence_pass": sum(has_anustubh_cadence(row["external_target"]) for row in output),
        "anustubh_one_draw_success": sum(row["pool"][0]["cadence_success"] for row in output),
        "anustubh_full_candidate_prompt_success": sum(any(x["cadence_success"] for x in row["pool"]) for row in output),
        "anustubh_prefix_prompt_success": sum(row["cadence_pick"] is not None for row in output),
    }
    count = min(len(row["pool"]) for row in output)
    candidates = [candidate for row in output for candidate in row["pool"]]
    picks = [row["cadence_selection"] for row in output if row["cadence_selection"]]
    summary["candidate_diagnostics"] = {
        "total": len(candidates),
        "eligible": sum(candidate["eligible"] for candidate in candidates),
        "scanner_failed": sum(candidate["external_pattern"] is None for candidate in candidates),
        "full_cadence_passes": sum(candidate["cadence_success"] for candidate in candidates),
        "selected_full": sum(pick["source"] == "full" for pick in picks),
        "selected_word_boundary": sum(pick["source"] == "word_boundary" for pick in picks),
        "selected_distinct_texts": len({devanagari_key(pick["text"]) for pick in picks}),
        "selected_adjacent_repeated_word": sum(
            any(a == b for a, b in zip(words, words[1:]))
            for pick in picks if (words := pick["text"].split())),
    }
    summary["combined_cadence_by_budget"] = {
        budget: sum(row["first_cadence_draw"] is not None and row["first_cadence_draw"] <= budget
                    for row in output)
        for budget in (1, 4, 8, 16, 32, 64, 128, 256) if budget <= count
    }
    args.output.write_text(json.dumps({"summary": summary, "rows": output}, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
