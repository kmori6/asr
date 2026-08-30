# LibriSpeech experiment

This experiment contains data preparation, training, evaluation, and inference workflows for the 16 kHz
[LibriSpeech corpus](https://www.openslr.org/12/). Run the commands below from `experiments/librispeech` after setting
up the repository environment.

## ASR data preparation

Download LibriSpeech, create the manifests, and train the shared tokenizer:

```bash
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

## Language-modeling data preparation

Download the normalized text from [OpenSLR SLR11](https://www.openslr.org/11/) and create reproducible training and
validation splits:

```bash
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
perplexity to `results/transformer_lm_evaluation/metrics.json`.

## FastConformer RNN-T

Train, evaluate, or run inference with the full-context model:

```bash
uv run scripts/train_fast_conformer_rnnt.py
uv run scripts/evaluate_fast_conformer_rnnt.py
uv run scripts/infer_fast_conformer_rnnt.py \
  infer.input_path=/path/to/audio.wav
```

The default configuration is `config/fast_conformer_rnnt.yaml`. Checkpoints, metric history, and loss plots are written
to `results/fast_conformer_rnnt/`. Set `train.checkpoint_path` in the configuration to resume from a checkpoint.

Train, evaluate, or run inference with the cache-aware streaming model:

```bash
uv run scripts/train_streaming_fast_conformer_rnnt.py
uv run scripts/evaluate_streaming_fast_conformer_rnnt.py
uv run scripts/infer_streaming_fast_conformer_rnnt.py \
  infer.input_path=/path/to/audio.wav
```

The streaming workflow uses `config/streaming_fast_conformer_rnnt.yaml` and writes checkpoints to
`results/streaming_fast_conformer_rnnt/`. Evaluation writes references, hypotheses, predictions, and WER metrics to
the model-specific evaluation directory. Inference prints the transcript and writes token IDs, the decoding score,
timing, and mode-specific settings to JSON.

## HuBERT CTC fine-tuning

Fine-tune Transformers' pretrained `facebook/hubert-base-ls960` checkpoint with a CTC head:

```bash
uv run scripts/train_hubert_ctc.py
uv run scripts/evaluate_hubert_ctc.py
uv run scripts/infer_hubert_ctc.py \
  infer.input_path=/path/to/audio.wav
```

The default configuration is `config/hubert_ctc.yaml`. Training writes the best model, processor, checkpoints, Trainer
state, and metrics to `results/hubert_ctc/`. Set `train.checkpoint_path` to a checkpoint directory to resume.
Evaluation writes predictions and corpus WER to `results/hubert_ctc_evaluation/`. Use
`evaluate.test_file=test-other.json` to select the other test split or set `evaluate.max_samples` to a positive integer
for a smoke test. Inference writes detailed results to `results/hubert_ctc_inference.json`.

## Whisper encoder-decoder fine-tuning

Fine-tune Transformers' `openai/whisper-small` checkpoint for English transcription:

```bash
uv run scripts/train_whisper.py
uv run scripts/evaluate_whisper.py
uv run scripts/infer_whisper.py \
  infer.input_path=/path/to/audio.wav
```

The default configuration is `config/whisper.yaml`. Training writes the best validation-loss model, processor,
checkpoints, Trainer state, and metrics to `results/whisper/`. Evaluation writes predictions and WER to
`results/whisper_evaluation/`, while inference writes detailed results to `results/whisper_inference.json`.

Inference audio is downmixed to mono and resampled to 16 kHz when necessary. Whisper inputs longer than its 30-second
short-form limit are recorded as skipped without running generation and are excluded from evaluation WER.
