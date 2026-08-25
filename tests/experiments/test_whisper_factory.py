import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "experiments" / "librispeech" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from whisper_factory import is_whisper_short_form, recognize_whisper, restore_english_spelling_normalizer  # noqa: E402


def test_is_whisper_short_form_includes_exact_feature_window() -> None:
    processor = cast(
        WhisperProcessor,
        SimpleNamespace(feature_extractor=SimpleNamespace(n_samples=480_000)),
    )

    assert is_whisper_short_form(torch.zeros(480_000), processor)
    assert not is_whisper_short_form(torch.zeros(480_001), processor)


def test_recognize_whisper_passes_only_total_max_length() -> None:
    feature_extractor = Mock(n_samples=16_000)
    feature_extractor.return_value = {
        "input_features": torch.zeros(1, 80, 3_000),
        "attention_mask": torch.ones(1, 3_000, dtype=torch.long),
    }
    processor = cast(
        WhisperProcessor,
        SimpleNamespace(
            feature_extractor=feature_extractor,
            tokenizer=SimpleNamespace(prefix_tokens=[1, 2, 3, 4]),
        ),
    )
    generate = Mock(return_value=torch.tensor([[5, 6]]))
    model = cast(
        WhisperForConditionalGeneration,
        SimpleNamespace(
            config=SimpleNamespace(max_target_positions=448),
            generate=generate,
        ),
    )

    token_ids = recognize_whisper(
        waveform=torch.zeros(8_000),
        sample_rate=16_000,
        model=model,
        processor=processor,
        language="english",
        task="transcribe",
        num_beams=1,
        max_new_tokens=256,
        device=torch.device("cpu"),
        amp_dtype=torch.float32,
    )

    assert token_ids == [5, 6]
    assert generate.call_args.kwargs["max_length"] == 260
    assert "max_new_tokens" not in generate.call_args.kwargs


def test_restore_english_spelling_normalizer_fills_missing_mapping() -> None:
    tokenizer = cast(WhisperTokenizer, SimpleNamespace(english_spelling_normalizer=None))
    fallback_tokenizer = cast(
        WhisperTokenizer,
        SimpleNamespace(english_spelling_normalizer={"colour": "color"}),
    )

    restore_english_spelling_normalizer(tokenizer, fallback_tokenizer)

    assert tokenizer.english_spelling_normalizer == {"colour": "color"}
