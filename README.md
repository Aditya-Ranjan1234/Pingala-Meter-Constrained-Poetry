# Pingala Meter-Constrained Poetry

Can a language model find more Sanskrit lines that scan correctly if it checks metre before a line is complete? We tested this on 64 held-out prompts using the same DistilGPT-2 checkpoint and a cap of 4,096 sampled tokens per prompt.

[Research paper (PDF)](paper/vimarsini_revised.pdf) · [Presentation (PowerPoint)](paper/Pingala_Metre_Research.pptx) · [Benchmark results](results/guided_reader_review/automatic_summary.json)

## What the experiment found

| Result on 64 held-out prompts | Generate and filter | Metre guided |
|---|---:|---:|
| Prompts with a cadence match | 29 | 40 |
| Total sampled tokens | 193,856 | 159,580 |
| Median sampled tokens on 21 shared successes | 1,952 | 589 |

Guidance found cadence matches on 11 more prompts. The paired exact test gave `p = 0.052`. Token counts measure sampling cost and do not measure elapsed time. A blinded LLM review gave both sets low grammar and meaning scores. A phoneme-based scanner agreed with chandas on the 36 reviewed lines. The study collected no human ratings.

## How metre guidance works

1. Both decoders sample continuations from the fine-tuned model.
2. At a word boundary, the guided decoder scans the prefix and drops it if its settled syllables violate the target cadence. The last two syllables remain provisional.
3. Both methods accept a complete line only when it passes the basic 16-syllable anuṣṭubh check and is absent from the normalized corpus. A prompt can exhaust its budget without a match.

The [guided decoder](src/guided_study.py) contains the paired experiment. The smaller [Prastāra demo](src/demo.py) illustrates exact light/heavy pattern filtering using a fixed phrase list.

## Repository contents

- [Paper source](paper/vimarsini_revised.tex) and [presentation](paper/Pingala_Metre_Research.pptx)
- [Benchmark summary](results/guided_reader_review/automatic_summary.json), [paired uncertainty](results/guided_reader_review/paired_uncertainty.json), and [LLM review summary](results/guided_reader_review/llm_paired_summary.json)
- [Source code](src), [tests](tests), and [data sources](data/README.md)

The public checkout includes research code, aggregate results, and publication files. Download input datasets separately. Trained weights, raw review sheets, and local environments stay outside Git.

## Try the phrase demo and run tests

From the repository root, run the following commands with Python 3.11.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m src.demo
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

## Re-run the Sanskrit experiment

Install a CUDA-enabled PyTorch build, then install the remaining packages from `requirements.txt`. Download the pinned Chandomitra splits listed in [data/README.md](data/README.md).

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
