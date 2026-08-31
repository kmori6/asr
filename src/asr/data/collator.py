from typing import cast

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    Wav2Vec2FeatureExtractor,
    WhisperFeatureExtractor,
)

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
    """Create padded waveform and token targets for CTC models."""

    def __init__(self, tokenizer: PreTrainedTokenizerFast, blank_token_id: int) -> None:
        if not 0 <= blank_token_id < len(tokenizer):
            raise ValueError(f"blank_token_id must be in [0, {len(tokenizer)}), but got {blank_token_id}")

        self.tokenizer = tokenizer
        self.blank_token_id = blank_token_id

    def __call__(self, samples: list[SpeechTextSample]) -> dict[str, torch.Tensor]:
        """Collate speech-recognition samples without adding BOS or EOS."""
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


class HubertCTCCollator:
    """Create padded HuBERT inputs and token labels for CTC training.

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


class EncoderDecoderCollator:
    """Create CTC targets and shifted autoregressive targets for speech."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        pad_token_id: int,
        max_target_length: int,
        ignore_index: int = -100,
    ) -> None:
        if not 0 <= pad_token_id < len(tokenizer):
            raise ValueError(f"pad_token_id must be in [0, {len(tokenizer)})")
        if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
            raise ValueError("tokenizer must define BOS and EOS tokens")
        if max_target_length <= 0:
            raise ValueError("max_target_length must be positive")
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        self.max_target_length = max_target_length
        self.ignore_index = ignore_index

    def __call__(self, samples: list[SpeechTextSample]) -> dict[str, torch.Tensor]:
        """Create a FastConformer-Transformer training batch."""
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
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        token_ids = cast(list[list[int]], encoded["input_ids"])
        bos_token_id = cast(int, self.tokenizer.bos_token_id)
        eos_token_id = cast(int, self.tokenizer.eos_token_id)
        if any(len(ids) < 2 or ids[0] != bos_token_id or ids[-1] != eos_token_id for ids in token_ids):
            raise ValueError("Every target sequence must begin with BOS and end with EOS")

        ctc_sequences = [torch.tensor(ids[1:-1], dtype=torch.long) for ids in token_ids]
        ctc_label_lengths = torch.tensor([sequence.shape[0] for sequence in ctc_sequences], dtype=torch.long)
        if torch.any(ctc_label_lengths == 0):
            raise ValueError("Every CTC target sequence must contain at least one token")
        ctc_labels = pad_sequence(
            ctc_sequences,
            batch_first=True,
            padding_value=self.pad_token_id,
        )

        decoder_sequences = [torch.tensor(ids[:-1], dtype=torch.long) for ids in token_ids]
        label_sequences = [torch.tensor(ids[1:], dtype=torch.long) for ids in token_ids]
        target_lengths = torch.tensor([sequence.shape[0] for sequence in decoder_sequences], dtype=torch.long)
        longest_target = int(target_lengths.max().item())
        if longest_target > self.max_target_length:
            raise ValueError(
                f"Target sequences must not exceed {self.max_target_length} tokens, but got {longest_target}."
            )

        decoder_input_ids = pad_sequence(
            decoder_sequences,
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        labels = pad_sequence(
            label_sequences,
            batch_first=True,
            padding_value=self.ignore_index,
        )
        positions = torch.arange(longest_target)
        decoder_attention_mask = positions[None, :] < target_lengths[:, None]
        return {
            "waveforms": waveforms,
            "waveform_lengths": waveform_lengths,
            "ctc_labels": ctc_labels,
            "ctc_label_lengths": ctc_label_lengths,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels,
        }


class WhisperCollator:
    """Create padded Whisper inputs and autoregressive decoder labels.

    Audio is converted to fixed-length log-Mel features by Whisper's feature
    extractor. Token padding is replaced by ``-100`` for cross-entropy loss,
    and the leading decoder-start token is removed because Whisper inserts it
    again when shifting labels to create ``decoder_input_ids``.
    """

    def __init__(
        self,
        feature_extractor: WhisperFeatureExtractor,
        tokenizer: PreTrainedTokenizerBase,
        sample_rate: int,
        decoder_start_token_id: int,
        max_target_length: int,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if feature_extractor.sampling_rate != sample_rate:
            raise ValueError(
                f"Feature extractor sample rate must be {sample_rate}, but got {feature_extractor.sampling_rate}."
            )
        if tokenizer.pad_token_id is None:
            raise ValueError("tokenizer must define a pad token")
        if decoder_start_token_id < 0:
            raise ValueError("decoder_start_token_id must be non-negative")
        if max_target_length < 1:
            raise ValueError("max_target_length must be positive")

        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.decoder_start_token_id = decoder_start_token_id
        self.max_target_length = max_target_length

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
            if sample.waveform.shape[0] > self.feature_extractor.n_samples:
                max_duration_seconds = self.feature_extractor.n_samples / self.sample_rate
                raise ValueError(
                    f"Whisper audio must not exceed {max_duration_seconds:g} seconds: {sample.utterance_id}"
                )

        waveforms = [sample.waveform.detach().cpu().numpy() for sample in samples]
        input_batch = self.feature_extractor(
            waveforms,
            sampling_rate=self.sample_rate,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        label_batch = self.tokenizer(
            [sample.text for sample in samples],
            add_special_tokens=True,
            padding=True,
            truncation=False,
            return_attention_mask=True,
            return_tensors="pt",
        )

        labels = cast(torch.Tensor, label_batch["input_ids"])
        label_attention_mask = cast(torch.Tensor, label_batch["attention_mask"])
        if labels.shape[1] == 0 or not labels[:, 0].eq(self.decoder_start_token_id).all():
            raise ValueError("Every Whisper label sequence must begin with decoder_start_token_id")

        labels = labels[:, 1:]
        label_attention_mask = label_attention_mask[:, 1:]
        if labels.shape[1] > self.max_target_length:
            raise ValueError(
                f"Whisper labels must not exceed {self.max_target_length} tokens, but got {labels.shape[1]}."
            )

        batch = {name: cast(torch.Tensor, value) for name, value in input_batch.items()}
        batch["labels"] = labels.masked_fill(label_attention_mask.ne(1), -100)
        return batch
