# Sanskrit data

The main experiment uses the [Chandomitra dataset](https://huggingface.co/datasets/sanganaka/anushtup) at revision `f563890505b104d09dc86a978a311e136a9b0155`. Save the [training split](https://huggingface.co/datasets/sanganaka/anushtup/resolve/f563890505b104d09dc86a978a311e136a9b0155/data/train-00000-of-00001.parquet) as `data/chandomitra-train.parquet`. Save the [test split](https://huggingface.co/datasets/sanganaka/anushtup/resolve/f563890505b104d09dc86a978a311e136a9b0155/data/test-00000-of-00001.parquet) as `data/chandomitra-test.parquet`. The repository does not redistribute the downloaded splits.

`sanskrit_metrical/` is the raw CC BY-SA 4.0 Sanskrit Metrical Poetry dataset from [dukeofmtnview/sanskrit-metrical-poetry](https://huggingface.co/datasets/dukeofmtnview/sanskrit-metrical-poetry). The repository contains 550 records: 50 verse examples and 500 composition-task records. The loader repairs the malformed JSON schema and keeps the 50 verse examples as exact text supervision.

`prepare_sanskrit.py --scale` can append the CC0 Sansar stream. It is intentionally opt-in because the published shard is large and network-dependent.
