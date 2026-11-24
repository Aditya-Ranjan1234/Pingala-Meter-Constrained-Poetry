"""Paired GPU comparison of word-boundary pruning and sample-and-filter."""
import json
import logging
import random
import re
import runpy
import sys
from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .audit_scansion import devanagari_key, has_anustubh_cadence

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "artifacts/chandomitra_distilgpt2"
CADENCE = {4: "L", 5: "G", 6: "G", 12: "L", 13: "G", 14: "L"}
LIMIT = 4096
BATCH = 16
MAX_LENGTH = 72


def viable(pattern):
    # A future onset can change the weight at the junction. Keep the
    # final two syllables uncommitted and re-scan after the next word.
    committed = pattern[:-2]
    return len(committed) <= 16 and all(
        committed[i] == weight for i, weight in CADENCE.items() if i < len(committed))


@torch.inference_mode()
def main():
    logging.disable(logging.CRITICAL)
    cache = ROOT / ".uv-cache/archive-v0"
    sys.path.extend(str(p) for p in sorted(cache.iterdir()) if p.is_dir())
    weights = runpy.run_path(str(next(cache.rglob("chandas/syllabize.py"))))["to_weight_list"]

    @lru_cache(maxsize=4096)
    def scan(text):
        if not re.fullmatch(r"[\u0900-\u0963 ]+", text):
            return ""
        try:
            return "".join(weights(text))
        except ValueError:
            return ""

    splits = json.loads((CHECKPOINT / "splits.json").read_text(encoding="utf-8"))
    corpus = {devanagari_key(line) for rows in splits.values() for line in rows}
    dev = json.loads((ROOT / "results/chandomitra_generation_128.json").read_text(encoding="utf-8"))
    seen = {devanagari_key(row["prompt"]) for row in dev["rows"]}
    references = {}
    for ref in splits["test"]:
        words = ref.split()
        if len(words) < 2:
            continue
        key = devanagari_key(words[0])
        if key not in seen and has_anustubh_cadence(scan(ref)):
            references.setdefault(key, (words[0], ref))
    prompts = random.Random(20261123).sample(list(references.values()), 64)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    model = AutoModelForCausalLM.from_pretrained(CHECKPOINT).cuda().eval()

    def search(prompt, guided, seed):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        start = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        used = discarded = 0
        while used < LIMIT:
            size = min(BATCH, LIMIT - used)
            ids = start.repeat(size, 1)
            active = list(range(size))
            boundaries = [""] * size
            past = None
            for step in range(MAX_LENGTH):
                if not active or used + len(active) > LIMIT:
                    break
                output = model(input_ids=ids if past is None else ids[:, -1:], past_key_values=past,
                               use_cache=True)
                past = output.past_key_values
                logits = output.logits[active, -1].float() / .8
                values, idx = logits.sort(descending=True)
                probabilities = values.softmax(-1)
                probabilities.masked_fill_(probabilities.cumsum(-1) - probabilities >= .9, 0)
                draws = idx.gather(1, torch.multinomial(probabilities, 1, generator=generator))
                tokens = torch.full((size, 1), tokenizer.eos_token_id, device="cuda", dtype=torch.long)
                tokens[active] = draws
                ids = torch.cat((ids, tokens), dim=1)
                used += len(active)
                decoded = tokenizer.batch_decode(ids[active], skip_special_tokens=True)
                kept = []
                for lane, raw, token in zip(active, decoded, draws.squeeze(1).tolist()):
                    head = re.split(r"[।॥\r\n]", raw, maxsplit=1)[0]
                    terminal = head != raw or token == tokenizer.eos_token_id or step == MAX_LENGTH - 1
                    complete = head.strip() if terminal or head.endswith(" ") else head.rpartition(" ")[0].strip()
                    complete = " ".join(complete.split())
                    if complete != boundaries[lane]:
                        boundaries[lane] = complete
                        pattern = scan(complete)
                        if (complete.startswith(prompt + " ") and has_anustubh_cadence(pattern)
                                and devanagari_key(complete) not in corpus):
                            return {"text": complete, "pattern": pattern, "tokens": used, "pruned": discarded}
                        if guided and pattern and not viable(pattern):
                            discarded += 1
                            continue
                    if not terminal:
                        kept.append(lane)
                active = kept
            if used + len(active) > LIMIT:
                # Spend the remaining budget on a smaller batch.
                continue
        return {"text": None, "pattern": None, "tokens": used, "pruned": discarded}

    rows = []
    for i, (prompt, reference) in enumerate(prompts):
        row = {"prompt": prompt, "reference": reference, "reference_pattern": scan(reference)}
        for name, guided in (("baseline", False), ("guided", True)):
            row[name] = search(prompt, guided, 6000 + i)
        rows.append(row)
        file = ROOT / "results/guided_fresh_64.json"
        file.write_text(json.dumps({"token_cap": LIMIT, "rows": rows}, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        print(i + 1, {name: (bool(row[name]["text"]), row[name]["tokens"])
                      for name in ("baseline", "guided")}, flush=True)


if __name__ == "__main__":
    main()
