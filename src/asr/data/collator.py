from typing import cast

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerFast

from asr.data.dataset import SpeechTextSample


class SpeechTextCollator:
    """Create a padded batch of waveforms and tokenized transcripts."""

    def __init__(self, tokenizer: PreTrainedTokenizerFast, blank_token_id: int) -> None:
        if not 0 <= blank_token_id < len(tokenizer):
            raise ValueError(f"blank_token_id must be in [0, {len(tokenizer)}), but got {blank_token_id}")

        self.tokenizer = tokenizer
        self.blank_token_id = blank_token_id

    def __call__(self, samples: list[SpeechTextSample]) -> dict[str, torch.Tensor]:
        """Collate speech-recognition samples.

        Args:
            samples: Non-empty list of speech and transcript samples.

        Returns:
            A batch containing ``waveforms`` with shape ``(batch, max_num_samples)``,
            ``waveform_lengths`` with shape ``(batch,)``, ``labels`` with shape
            ``(batch, max_label_length)``, and ``label_lengths`` with shape ``(batch,)``.
            Waveforms are padded with zero and labels with ``blank_token_id``.
        """
        if not samples:
            raise ValueError("samples must not be empty")

        waveform_lengths = torch.tensor([sample.waveform.shape[0] for sample in samples], dtype=torch.long)
        waveforms = pad_sequence(
            [sample.waveform for sample in samples],
            batch_first=True,
            padding_value=0.0,
        )

        encoded = self.tokenizer(
            [sample.text for sample in samples],
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        token_ids = cast(list[list[int]], encoded["input_ids"])
        label_sequences = [torch.tensor(ids, dtype=torch.long) for ids in token_ids]
        label_lengths = torch.tensor([label.shape[0] for label in label_sequences], dtype=torch.long)
        labels = pad_sequence(
            label_sequences,
            batch_first=True,
            padding_value=self.blank_token_id,
        )

        return {
            "waveforms": waveforms,
            "waveform_lengths": waveform_lengths,
            "labels": labels,
            "label_lengths": label_lengths,
        }
