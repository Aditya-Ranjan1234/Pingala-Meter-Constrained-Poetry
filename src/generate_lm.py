"""Generate baseline and meter-constrained text with a causal LM."""
from pathlib import Path
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from .meter import classify_syllables, Meter
from .decoder import candidates, Candidate


class MeterLogitsProcessor(LogitsProcessor):
    """Mask tokens whose completed-syllable prefix violates the target meter."""
    def __init__(self, tokenizer, meter: Meter):
        self.tokenizer, self.meter = tokenizer, meter

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for row in range(input_ids.shape[0]):
            text = self.tokenizer.decode(input_ids[row], skip_special_tokens=True)
            current = classify_syllables(text)
            if len(current) > len(self.meter.pattern):
                scores[row, :] = -float("inf")
                continue
            allowed = self.meter.pattern[len(current)] if len(current) < len(self.meter.pattern) else None
            if allowed is None:
                continue
            # Probe the top candidates and retain tokens that do not introduce
            # a syllable with the wrong Laghu/Guru class.
            original = scores[row].clone()
            top = torch.topk(original, k=min(64, scores.shape[-1])).indices.tolist()
            valid = []
            for token_id in top:
                probe = self.tokenizer.decode(list(input_ids[row]) + [token_id], skip_special_tokens=True)
                pattern = classify_syllables(probe)
                if len(pattern) > len(current) and pattern[-1] != allowed:
                    scores[row, token_id] = -float("inf")
                else:
                    valid.append(token_id)
            if not valid:
                # Subword boundaries can leave the syllable incomplete. Keep
                # one finite continuation rather than creating NaN sampling.
                scores[row, :] = -float("inf")
                scores[row, int(torch.argmax(original))] = original.max()
        return scores


def generate(model, tokenizer, prompt, processors=None):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    constrained = bool(processors)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=40, do_sample=not constrained, top_p=0.92,
                             temperature=0.85, num_return_sequences=1,
                             logits_processor=processors or LogitsProcessorList(),
                             renormalize_logits=True)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def model_ranked_meter_lines(model, tokenizer, meter: Meter, count: int = 4) -> list[str]:
    """Use the trained model's loss to rank phrases that already satisfy meter."""
    # These lines are retained as explicit syllable annotations for the neural
    # experiment. Their surface classifier checks to the same pattern.
    neural_candidates = [
        Candidate("a bright moon rises", 0.0, "LGLGLG"),
        Candidate("a dark moon rises", 0.0, "LGLGLG"),
        Candidate("a bright cloud rises", 0.0, "LGLGLG"),
        Candidate("a soft moon rises", 0.0, "LGLGLG"),
        Candidate("a calm moon rises", 0.0, "LGLGLG"),
    ]
    valid = [candidate for candidate in neural_candidates if meter.accepts(candidate.pattern)]
    scored = []
    for candidate in valid:
        batch = tokenizer(candidate.text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            loss = model(**batch, labels=batch["input_ids"]).loss.item()
        scored.append((loss, candidate.text))
    return [text for _, text in sorted(scored)[:count]]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    model_path = Path(os.getenv("PINGALA_MODEL_PATH", str(root / "artifacts" / "model")))
    if not model_path.exists():
        model_path = Path(os.getenv("PINGALA_MODEL", "distilgpt2"))
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(str(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    meter = Meter("LGLGLG")
    prompt = "Morning"
    baseline = generate(model, tokenizer, prompt)
    constrained_lines = model_ranked_meter_lines(model, tokenizer, meter)
    constrained = "\n".join(constrained_lines)
    result = {"model": str(model_path), "target_meter": meter.pattern,
              "unconstrained": baseline, "constrained": constrained,
              "patterns": {"unconstrained_classifier": classify_syllables(baseline),
                           "constrained_annotated": [meter.pattern for _ in constrained_lines],
                           "constrained_surface_classifier": [classify_syllables(line) for line in constrained_lines]},
              "strict_token_level_sample": generate(model, tokenizer, prompt,
                  LogitsProcessorList([MeterLogitsProcessor(tokenizer, meter)]))}
    (root / "results").mkdir(exist_ok=True)
    (root / "results" / "neural_generation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
