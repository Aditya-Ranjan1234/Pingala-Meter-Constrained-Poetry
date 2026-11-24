"""Compare token-weighted loss on the same held-out Chandomitra lines."""
import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("splits", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("models", type=Path, nargs="+")
    args = parser.parse_args()
    lines = json.loads(args.splits.read_text(encoding="utf-8"))["test"]
    results = []
    for checkpoint in args.models:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(checkpoint).cuda().eval()
        nll = 0.0
        tokens = 0
        for start in range(0, len(lines), 16):
            batch = tokenizer(lines[start:start + 16], padding=True, truncation=True,
                              max_length=96, return_tensors="pt").to("cuda")
            labels = batch.input_ids.masked_fill(batch.attention_mask == 0, -100)
            count = int((labels[:, 1:] != -100).sum())
            if not count:
                continue
            loss = model(**batch, labels=labels, use_cache=False).loss
            nll += float(loss) * count
            tokens += count
        result = {"checkpoint": str(checkpoint), "lines": len(lines), "tokens": tokens,
                  "loss": nll / tokens, "perplexity": math.exp(nll / tokens)}
        results.append(result)
        print(result, flush=True)
        del model
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
