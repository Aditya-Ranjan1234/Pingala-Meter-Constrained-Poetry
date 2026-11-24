# Pingala Meter-Constrained Poetry

This project measures whether checking Sanskrit metre at word boundaries helps a language model find lines with a basic anuṣṭubh cadence. Both decoders used the same fine-tuned DistilGPT-2 checkpoint and a cap of 4,096 sampled tokens per prompt.

| Result on 64 held-out prompts | Generate and filter | Metre guided |
|---|---:|---:|
| Prompts with a cadence match | 29 | 40 |
| Total sampled tokens | 193,856 | 159,580 |
| Median sampled tokens on 21 shared successes | 1,952 | 589 |

A blinded LLM review scored both sets of generated lines low for grammar and meaning. A separate phoneme-based scanner agreed with chandas on 36 selected lines. Read the [paper](paper/vimarsini_revised.pdf) for limits on what these results show.

## Repository contents

- [Paper source](paper/vimarsini_revised.tex) and [presentation](paper/Pingala_Metre_Research.pptx)
- [Benchmark summary](results/guided_reader_review/automatic_summary.json), [paired uncertainty](results/guided_reader_review/paired_uncertainty.json), and [LLM review summary](results/guided_reader_review/llm_paired_summary.json)
- [Source code](src), [tests](tests), and [data sources](data/README.md)

The public checkout includes research code, aggregate results, and publication files. Download input datasets separately. Trained weights, raw review sheets, and local environments stay outside Git.

## Run the tests

From the repository root, run the following commands with Python 3.11.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

## Re-run the Sanskrit experiment

Use a CUDA-capable PyTorch installation. Install the remaining packages from `requirements.txt`. Download the pinned Chandomitra splits listed in [data/README.md](data/README.md).

```powershell
$env:PINGALA_VERSE_SPLIT = 'chandomitra'
$env:PINGALA_OUTPUT = 'artifacts/chandomitra_distilgpt2'
$env:PINGALA_EPOCHS = '3'
$env:PINGALA_BATCH_SIZE = '8'
.venv/Scripts/python.exe -m src.train_lm
.venv/Scripts/python.exe -m src.evaluate_generation artifacts/chandomitra_distilgpt2 results/chandomitra_generation_128.json --prompts 64 --samples 128
.venv/Scripts/python.exe -m src.guided_study
```

The last command uses the local chandas scanner and its dependencies from `.uv-cache/archive-v0`. See `src/guided_study.py` for the scanner entry point. The run writes its prompt-level trace to `results/guided_fresh_64.json`. The original traces are local, so a fresh checkout can inspect the published summaries or regenerate traces by running the experiment.

## Build the manuscript

Compile `paper/vimarsini_revised.tex` with XeLaTeX or Tectonic. Keep `paper/review-references.tex` next to the source and install Times New Roman and Nirmala UI, or select equivalent fonts in the preamble. The source identifies IAST Roman and Unicode Devanagari using distinct macros. The author-date reference list is a submission preview for the journal's production workflow.
