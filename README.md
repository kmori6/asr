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

## LibriSpeech language-modeling data preparation

Download the normalized text from [OpenSLR SLR11](https://www.openslr.org/11/) and create reproducible training and
validation splits:

```bash
cd experiments/librispeech

bash scripts/download_librispeech_lm.sh data/lm

uv run scripts/prepare_lm_dataset.py \
  --input_path data/lm/librispeech-lm-norm.txt.gz \
  --out_dir data/lm
```

This writes `data/lm/train.json` and `data/lm/valid.json` as JSON Lines. Each record contains one normalized `text`
field. ASR and LM use the shared tokenizer created above. ASR tokenization omits `[BOS]` and `[EOS]`, while LM
tokenization uses them as sentence boundaries.

Train the causal Transformer LM and evaluate validation perplexity:

```bash
uv run scripts/train_transformer_lm.py

uv run scripts/evaluate_transformer_lm.py
```

The default configuration is `config/transformer_lm.yaml`. Training writes the best model, tokenizer, Trainer state,
metrics, and resolved configuration to `results/transformer_lm/`. Evaluation writes token-level validation loss and
perplexity to `results/transformer_lm_evaluation/metrics.json`. The model's cached `predict` method is intended for
future shallow-fusion and rescoring integration with the ASR decoders.

## FastConformer RNN-T

The LibriSpeech experiment provides separate non-streaming and streaming workflows. After preparing the manifests and
tokenizer, train the full-context model from the experiment directory:

```bash
cd experiments/librispeech

uv run scripts/train_fast_conformer_rnnt.py
```

The default configuration is `config/fast_conformer_rnnt.yaml`. Checkpoints, metric history, and loss plots are written
to `results/fast_conformer_rnnt/`. Set `train.checkpoint_path` in the configuration to resume from the latest checkpoint.

Evaluate the saved checkpoint or recognize an arbitrary audio file with full-context encoding:

```bash
uv run scripts/evaluate_fast_conformer_rnnt.py

uv run scripts/infer_fast_conformer_rnnt.py \
  infer.input_path=/path/to/audio.wav
```

Train and run the cache-aware streaming model with the corresponding streaming entrypoints:

```bash
uv run scripts/train_streaming_fast_conformer_rnnt.py

uv run scripts/evaluate_streaming_fast_conformer_rnnt.py

uv run scripts/infer_streaming_fast_conformer_rnnt.py \
  infer.input_path=/path/to/audio.wav
```

The streaming workflow uses `config/streaming_fast_conformer_rnnt.yaml` and writes checkpoints to
`results/streaming_fast_conformer_rnnt/`. Evaluation writes references, hypotheses, predictions, and WER metrics to
the model-specific evaluation directory. Inference prints the transcript and writes token IDs, the decoding score,
timing, and mode-specific settings to JSON. Input audio is downmixed to mono and resampled to 16 kHz when necessary.

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

## Whisper encoder-decoder fine-tuning

The encoder-decoder experiment fine-tunes Transformers' `openai/whisper-small` checkpoint for English LibriSpeech
transcription. It uses Whisper's own processor and the repository's `EncoderDecoderCollator`:

```bash
cd experiments/librispeech

uv run scripts/train_whisper.py
```

The default configuration is `config/whisper.yaml`. The best validation-loss model, processor, checkpoints, trainer
state, and train/validation metrics are written to `results/whisper/`.

Evaluate the saved model with normalized English WER:

```bash
uv run scripts/evaluate_whisper.py
```

Evaluation artifacts are written to `results/whisper_evaluation/`. Utterances longer than Whisper's 30-second
short-form limit are recorded as skipped and excluded from WER. Recognize an audio file with:

```bash
uv run scripts/infer_whisper.py \
  infer.input_path=/path/to/audio.wav
```

The transcript is printed to standard output and detailed inference metadata is written to
`results/whisper_inference.json`. Input audio is downmixed to mono and resampled to 16 kHz when necessary; audio longer
than 30 seconds is recorded as skipped without running generation.
