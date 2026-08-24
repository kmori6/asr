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

## FastConformer RNN-T training

After preparing the manifests and tokenizer, start training from the LibriSpeech experiment directory:

```bash
cd experiments/librispeech

uv run scripts/train_fast_conformer_rnnt.py
```

The default configuration is `config/fast_conformer_rnnt.yaml`. Checkpoints, metric history, and loss plots are written
to `results/fast_conformer_rnnt/`. Set `train.checkpoint_path` in the configuration to resume from the latest checkpoint.

Recognize an arbitrary audio file with the trained checkpoint:

```bash
cd experiments/librispeech

uv run scripts/infer_fast_conformer.py \
  infer.input_path=/path/to/audio.wav
```

The transcript is printed to standard output and a JSON record containing the transcript, token IDs, decoding score,
timing, and inference settings is written to `results/inference.json`. Input audio is downmixed to mono and resampled
to 16 kHz when necessary. Set `infer.streaming=false` for complete-waveform inference.

## HuBERT CTC fine-tuning

The HuBERT experiment reuses the prepared manifests and tokenizer, and fine-tunes Transformers' pretrained
`facebook/hubert-base-ls960` checkpoint with a CTC head:

```bash
cd experiments/librispeech

uv run scripts/train_hubert_ctc.py
```

The default configuration is `config/hubert_ctc.yaml`. The best model, processor, checkpoints, trainer state, and
train/validation metrics are written to `results/hubert_ctc/`. Set `train.checkpoint_path` to a checkpoint directory
such as `results/hubert_ctc/checkpoint-1000` to resume.

Evaluate the saved model with the CTC prefix beam search implementation:

```bash
uv run scripts/evaluate_hubert_ctc.py
```

References, hypotheses, per-utterance predictions, and corpus WER are written to
`results/hubert_ctc_evaluation/`. Use `evaluate.test_file=test-other.json` to select the other test split, or set
`evaluate.max_samples` to a positive integer for a short smoke test.

Recognize an arbitrary audio file with the same decoder:

```bash
uv run scripts/infer_hubert_ctc.py \
  infer.input_path=/path/to/audio.wav
```

The transcript is printed to standard output, while token IDs, the length-normalized beam-search score, timing, and
input metadata are written to `results/hubert_ctc_inference.json`. Input audio is downmixed to mono and resampled to
16 kHz when necessary.
