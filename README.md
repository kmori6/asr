# asr

A research toolkit for automatic speech recognition. Reusable implementations live under `src/asr`, while
dataset-specific workflows live under `experiments/`.

## Architecture

- Frontend
  - Log-mel spectrogram
  - SpecAugment
- CTC (full-context and streaming)
- RNN-T (full-context and streaming)
- Encoder-decoder
- Encoder-LLM
- Decoding
  - CTC beam search
  - RNN-T beam search
  - Joint CTC/attention beam search
  - Shallow fusion

## Repository layout

```text
src/asr/                 # reusable models, modules, decoding, data, and training code
experiments/librispeech/ # LibriSpeech configuration and runnable workflows
tests/                   # unit and integration tests for the reusable library
```

See the [LibriSpeech experiment](experiments/librispeech/README.md) for data preparation, training, evaluation, and
inference commands.

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --group dev
```

## Development

Run the complete test suite or a focused test file from the repository root:

```bash
uv run pytest
uv run pytest tests/decoding/test_ctc_beam_search.py
```

Run the repository checks before submitting a change:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy --ignore-missing-imports .
uv run pytest
```
