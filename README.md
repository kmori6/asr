# asr

A research repository for automatic speech recognition. Reusable implementations live under `src/asr`, while
dataset-specific workflows are organized under `experiments/`.

## Setup

```bash
uv sync --group dev
```

## LibriSpeech data preparation

The initial experiment uses the 16 kHz LibriSpeech corpus from [OpenSLR](https://www.openslr.org/12/).

```bash
cd experiments/librispeech

bash scripts/download_librispeech.sh data

uv run scripts/prepare_dataset.py \
  --data_dir data \
  --out_dir data \
  --min_duration 1.0 \
  --max_duration 20.0

uv run scripts/train_tokenizer.py \
  --text_path data/train.txt \
  --out_dir results/tokenizer \
  --vocab_size 128
```

The preparation script writes `train.json`, `valid.json`, `test-clean.json`, `test-other.json`, and `train.txt` under
`data/`. Generated data and results are not committed.
