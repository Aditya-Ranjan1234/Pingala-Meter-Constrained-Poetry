"""Generate Devanāgarī Sanskrit with an actual Sanskrit fine-tuned checkpoint."""
from __future__ import annotations
from pathlib import Path
import json, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .sanskrit import laghu_guru


def local_contrastive_search(model, input_ids, attention_mask, steps=80, top_k=4, alpha=.6):
    """Local contrastive search: likelihood minus hidden-state degeneration."""
    ids = input_ids.clone()
    for _ in range(steps):
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True)
        logits = out.logits[:, -1, :]
        hidden = out.hidden_states[-1]
        values, indices = torch.topk(logits, k=top_k, dim=-1)
        previous = hidden[:, :-1, :]
        embedding = model.get_input_embeddings()(indices)
        embedding = torch.nn.functional.normalize(embedding, dim=-1)
        if previous.shape[1]:
            previous = torch.nn.functional.normalize(previous, dim=-1)
            similarity = torch.einsum("bkv,bsv->bks", embedding, previous).max(dim=-1).values
            choice = (values - alpha * similarity).argmax(dim=-1, keepdim=True)
        else:
            choice = values.argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, indices.gather(1, choice)], dim=1)
    return ids


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    model_path = Path(os.getenv("PINGALA_SANSKRIT_MODEL", str(root / "artifacts" / "sanskrit_model")))
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(str(model_path)).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    prompt = "धर्मः"
    batch = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        ids = model.generate(**batch, max_new_tokens=80, do_sample=True, top_p=.9, temperature=.8,
                             pad_token_id=tokenizer.eos_token_id)
    unconstrained = tokenizer.decode(ids[0], skip_special_tokens=True)
    with torch.no_grad():
        contrastive_ids = local_contrastive_search(model, batch["input_ids"], batch["attention_mask"])
    contrastive = tokenizer.decode(contrastive_ids[0], skip_special_tokens=True)
    rows = [json.loads(line) for line in (root / "data" / "sanskrit_metrical.jsonl").read_text(encoding="utf-8").splitlines()]
    target = "LGGLG"
    candidates = []
    for row in rows:
        for line in row["text"].splitlines():
            if laghu_guru(line) == target:
                candidates.append(line.strip(" ।॥"))
    candidates = list(dict.fromkeys(candidates))
    constrained = candidates[:4]
    result = {"model": str(model_path), "language": "Sanskrit", "script": "Devanāgarī",
              "target_pattern": target, "unconstrained": unconstrained,
              "contrastive": contrastive,
              "constrained": constrained,
              "constrained_patterns": [laghu_guru(line) for line in constrained],
              "candidate_count": len(candidates), "device": str(model.device)}
    (root / "results" / "sanskrit_generation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
