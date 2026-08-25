import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast, Wav2Vec2FeatureExtractor, WhisperFeatureExtractor

from asr.data import CTCCollator, EncoderDecoderCollator, RNNTCollator, SpeechTextSample


def create_tokenizer() -> PreTrainedTokenizerFast:
    backend_tokenizer = Tokenizer(
        WordLevel(
            vocab={"[BLANK]": 0, "[UNK]": 1, "hello": 2, "world": 3},
            unk_token="[UNK]",
        )
    )
    backend_tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend_tokenizer,
        unk_token="[UNK]",
        additional_special_tokens=["[BLANK]"],
    )
    return tokenizer


def create_samples() -> list[SpeechTextSample]:
    return [
        SpeechTextSample(
            utterance_id="1",
            waveform=torch.tensor([0.1, 0.2, 0.3]),
            sample_rate=16_000,
            text="hello world",
        ),
        SpeechTextSample(
            utterance_id="2",
            waveform=torch.tensor([0.4, 0.5]),
            sample_rate=16_000,
            text="hello",
        ),
    ]


def test_rnnt_collator_with_dataloader() -> None:
    tokenizer = create_tokenizer()
    samples = create_samples()
    # DataLoader's generic type does not model collate_fn changing the item type into a batch mapping.
    dataloader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        samples,  # type: ignore[arg-type]
        batch_size=2,
        shuffle=False,
        collate_fn=RNNTCollator(tokenizer, blank_token_id=0),
    )

    batch = next(iter(dataloader))

    torch.testing.assert_close(batch["waveforms"], torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.0]]))
    torch.testing.assert_close(batch["waveform_lengths"], torch.tensor([3, 2]))
    torch.testing.assert_close(batch["labels"], torch.tensor([[2, 3], [2, 0]]))
    torch.testing.assert_close(batch["label_lengths"], torch.tensor([2, 1]))


def test_ctc_collator_pads_inputs_and_masks_label_padding() -> None:
    tokenizer = create_tokenizer()
    tokenizer.pad_token = "[BLANK]"
    feature_extractor = Wav2Vec2FeatureExtractor(
        sampling_rate=16_000,
        do_normalize=False,
        return_attention_mask=True,
    )
    collator = CTCCollator(feature_extractor, tokenizer, sample_rate=16_000)

    batch = collator(create_samples())

    torch.testing.assert_close(batch["input_values"], torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.0]]))
    torch.testing.assert_close(batch["attention_mask"], torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.int32))
    torch.testing.assert_close(batch["labels"], torch.tensor([[2, 3], [2, -100]]))


def test_encoder_decoder_collator_creates_whisper_inputs_and_labels() -> None:
    backend_tokenizer = Tokenizer(
        WordLevel(
            vocab={"[PAD]": 0, "[UNK]": 1, "[SOT]": 2, "[EOS]": 3, "hello": 4, "world": 5},
            unk_token="[UNK]",
        )
    )
    backend_tokenizer.pre_tokenizer = Whitespace()
    backend_tokenizer.post_processor = TemplateProcessing(
        single="[SOT] $A [EOS]",
        special_tokens=[("[SOT]", 2), ("[EOS]", 3)],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend_tokenizer,
        pad_token="[PAD]",
        unk_token="[UNK]",
        bos_token="[SOT]",
        eos_token="[EOS]",
    )
    collator = EncoderDecoderCollator(
        feature_extractor=WhisperFeatureExtractor(sampling_rate=16_000),
        tokenizer=tokenizer,
        sample_rate=16_000,
        decoder_start_token_id=2,
        max_target_length=8,
    )

    batch = collator(create_samples())

    assert batch["input_features"].shape == (2, 80, 3_000)
    assert batch["attention_mask"].shape == (2, 3_000)
    torch.testing.assert_close(batch["labels"], torch.tensor([[4, 5, 3], [4, 3, -100]]))
