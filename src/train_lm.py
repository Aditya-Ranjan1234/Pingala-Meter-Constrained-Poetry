"""Fine-tune a small causal LM on the local poetry corpus.

The default model is distilgpt2. Set PINGALA_MODEL to another compatible
causal language model when a different size or language is required.
"""
from pathlib import Path
import os
import json
import random
import re
import unicodedata
import torch
from datasets import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    model_name = os.getenv("PINGALA_MODEL", "distilgpt2")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment. Install the CUDA PyTorch wheel and check nvidia-smi.")
    print(f"training on {torch.cuda.get_device_name(0)}")
    output = Path(os.getenv("PINGALA_OUTPUT", str(root / "artifacts" / "model")))
    corpus_path = Path(os.getenv("PINGALA_TRAIN_FILE", str(root / "data" / "train.txt")))
    lines = [x.strip() for x in corpus_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    evaluation = {}
    split_mode = os.getenv("PINGALA_VERSE_SPLIT")
    if split_mode in {"1", "chandomitra"}:
        if split_mode == "chandomitra":
            from pyarrow.parquet import read_table

            def read_verses(split):
                return read_table(root / "data" / f"chandomitra-{split}.parquet",
                                  columns=["Sanskrit"])["Sanskrit"].to_pylist()

            train_verses = read_verses("train")
            test_verses = read_verses("test")
            random.Random(42).shuffle(train_verses)
            validation_verses, train_verses = train_verses[:500], train_verses[500:]
        else:
            verses = [json.loads(row)["text"] for row in
                      (root / "data" / "sanskrit_metrical.jsonl").read_text(encoding="utf-8").splitlines()]
            random.Random(42).shuffle(verses)
            test_verses, validation_verses, train_verses = verses[:10], verses[10:20], verses[20:]

        def distinct_lines(rows):
            if split_mode == "chandomitra":
                lines = {unicodedata.normalize("NFC", " ".join(line.split()))
                         for verse in rows if verse
                         for line in re.split(r"[।॥\r\n|]+", verse)}
                return sorted(line for line in lines if re.search(r"[\u0905-\u0939]", line))
            return sorted({line.strip() for verse in rows for line in verse.splitlines() if line.strip()})

        def text_key(line):
            if split_mode == "chandomitra":
                return re.sub(r"[^\u0900-\u0963]", "", line)
            return line

        test = distinct_lines(test_verses)
        test_keys = {text_key(line) for line in test}
        validation = [line for line in distinct_lines(validation_verses) if text_key(line) not in test_keys]
        held_out = test_keys | {text_key(line) for line in validation}
        lines = [line for line in distinct_lines(train_verses) if text_key(line) not in held_out]
        for name, split in [("train", lines), ("validation", validation), ("test", test)]:
            if not split:
                raise ValueError(f"Empty {name} split")
            evaluation[name] = split
        output.mkdir(parents=True, exist_ok=True)
        (output / "splits.json").write_text(json.dumps(evaluation, ensure_ascii=False), encoding="utf-8")
    print(f"training records: {len(lines)}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = Dataset.from_dict({"text": lines})
    tokenized = dataset.map(lambda batch: tokenizer(batch["text"], truncation=True, max_length=96), batched=True, remove_columns=["text"])
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    def tokenize(split):
        return Dataset.from_dict({"text": split}).map(
            lambda batch: tokenizer(batch["text"], truncation=True, max_length=96),
            batched=True, remove_columns=["text"])
    args = TrainingArguments(output_dir=str(output), num_train_epochs=int(os.getenv("PINGALA_EPOCHS", "3")),
                             per_device_train_batch_size=int(os.getenv("PINGALA_BATCH_SIZE", "2")),
                             learning_rate=5e-5, weight_decay=0.01,
                             logging_steps=25 if split_mode == "chandomitra" else 1,
                             save_strategy="epoch", report_to="none",
                             fp16=True, tf32=True, dataloader_pin_memory=True,
                             eval_strategy="epoch" if evaluation else "no",
                             load_best_model_at_end=bool(evaluation), save_total_limit=2,
                             save_only_model=True)
    trainer = Trainer(model=model, args=args, train_dataset=tokenized,
                      eval_dataset=tokenize(evaluation["validation"]) if evaluation else None,
                      data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False))
    trainer.train()
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    if evaluation:
        test_metrics = trainer.evaluate(tokenize(evaluation["test"]), metric_key_prefix="test")
        report = {"history": trainer.state.log_history, "best_step_checkpoint": trainer.state.best_model_checkpoint,
                  "test": test_metrics, "records": {key: len(value) for key, value in evaluation.items()}}
        (output / "heldout_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved fine-tuned model to {output}")


if __name__ == "__main__":
    main()
