"""Evaluate neural candidates and whole-line meter rejection on held-out prompts."""
import argparse
import json
from pathlib import Path
import re
import random
import unicodedata

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from .generate_sanskrit import local_contrastive_search
from .sanskrit import laghu_guru


def normalize(text):
    return "".join(ch for ch in unicodedata.normalize("NFC", text)
                   if "\u0900" <= ch <= "\u097f" and ch not in "।॥")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prompts", type=int, default=24)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--model", type=Path, help="Use another checkpoint with the same held-out split")
    args = parser.parse_args()
    set_seed(42)
    splits = json.loads((args.checkpoint / "splits.json").read_text(encoding="utf-8"))
    train = {normalize(line) for line in splits["train"]}
    model_path = args.model or args.checkpoint
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).cuda().eval()
    pairs = {}
    for line in splits["test"]:
        words = re.findall(r"[\u0900-\u0963]+", line)
        pattern = laghu_guru(line)
        if len(words) > 1 and len(pattern) >= 3:
            pairs.setdefault(words[0], pattern)
    rows = []
    population = list(pairs.items())
    examples = (random.Random(42).sample(population, min(args.prompts, len(population)))
                if args.prompts != 24 else population[:24])
    for prompt, target in examples:
        batch = tokenizer(prompt, return_tensors="pt").to(model.device)
        candidates = []
        for start in range(0, args.samples, 32):
            candidates.extend(model.generate(
                **batch, max_new_tokens=48, do_sample=True, top_p=.9, temperature=.8,
                num_return_sequences=min(32, args.samples - start),
                pad_token_id=tokenizer.eos_token_id))
        contrastive = local_contrastive_search(model, batch.input_ids, batch.attention_mask, steps=48)

        def score(ids):
            decoded = tokenizer.decode(ids, skip_special_tokens=True)
            text = re.split(r"[।॥\r\n]", decoded, maxsplit=1)[0].strip()
            pattern = laghu_guru(text)
            return {"text": text, "pattern": pattern, "match": pattern == target,
                    "added_text": text.startswith(prompt) and bool(text[len(prompt):].strip()),
                    "train_copy": normalize(text) in train}

        pool = [score(ids) for ids in candidates]
        selected = next((i for i, row in enumerate(pool)
                         if row["added_text"] and row["match"] and not row["train_copy"]), None)
        rows.append({"prompt": prompt, "target": target, "pool": pool,
                     "contrastive": score(contrastive[0]), "selected_index": selected})
        if len(rows) % 16 == 0:
            print(f"Completed {len(rows)} of {len(examples)} prompts", flush=True)
    def eligible(row):
        return row["added_text"] and row["match"] and not row["train_copy"]
    summary = {"prompts": len(rows), "sampled_candidates": sum(len(row["pool"]) for row in rows),
               "one_draw_success": sum(eligible(row["pool"][0]) for row in rows),
               "contrastive_success": sum(eligible(row["contrastive"]) for row in rows),
               "pool_candidate_success": sum(eligible(candidate) for row in rows for candidate in row["pool"]),
               "rejection_prompt_success": sum(row["selected_index"] is not None for row in rows)}
    args.output.write_text(json.dumps({"seed": 42, "checkpoint": str(model_path), "summary": summary, "rows": rows},
                                      ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
