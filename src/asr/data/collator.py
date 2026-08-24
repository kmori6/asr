from typing import cast

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast, Wav2Vec2FeatureExtractor

from asr.data.dataset import SpeechTextSample


class RNNTCollator:
    """Create a padded RNN-T batch of waveforms and tokenized transcripts."""

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


class CTCCollator:
    """Dynamically pad raw speech and token labels for CTC training.

    The feature extractor controls whether an input ``attention_mask`` is
    returned. Label padding is always replaced by ``-100``, which Transformers
    CTC models exclude from their loss.
    """

    def __init__(
        self,
        feature_extractor: Wav2Vec2FeatureExtractor,
        tokenizer: PreTrainedTokenizerBase,
        sample_rate: int,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if feature_extractor.sampling_rate != sample_rate:
            raise ValueError(
                f"Feature extractor sample rate must be {sample_rate}, but got {feature_extractor.sampling_rate}."
            )
        if tokenizer.pad_token_id is None:
            raise ValueError("tokenizer must define a pad token for the CTC blank")

        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate

    def __call__(self, samples: list[SpeechTextSample]) -> dict[str, torch.Tensor]:
        """Collate a non-empty list of mono waveform and transcript samples."""
        if not samples:
            raise ValueError("samples must not be empty")
        for sample in samples:
            if sample.sample_rate != self.sample_rate:
                raise ValueError(
                    f"Expected a {self.sample_rate} Hz sample rate, but got {sample.sample_rate} "
                    f"for {sample.utterance_id}."
                )

        # Wav2Vec2FeatureExtractor does not reliably accept a heterogeneous list
        # of torch tensors, so convert the CPU dataset tensors individually.
        waveforms = [sample.waveform.detach().cpu().numpy() for sample in samples]
        input_batch = self.feature_extractor(
            waveforms,
            sampling_rate=self.sample_rate,
            padding=True,
            return_tensors="pt",
        )
        label_batch = self.tokenizer(
            [sample.text for sample in samples],
            add_special_tokens=False,
            padding=True,
            truncation=False,
            return_attention_mask=True,
            return_tensors="pt",
        )

        labels = cast(torch.Tensor, label_batch["input_ids"])
        label_attention_mask = cast(torch.Tensor, label_batch["attention_mask"])
        if labels.shape[1] == 0:
            raise ValueError("CTC transcripts must produce at least one token")

        batch = {name: cast(torch.Tensor, value) for name, value in input_batch.items()}
        batch["labels"] = labels.masked_fill(label_attention_mask.ne(1), -100)
        return batch
