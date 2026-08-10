import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from asr.data import SpeechTextCollator, SpeechTextSample


def test_speech_text_collator_with_dataloader() -> None:
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
    samples = [
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
    # DataLoader's generic type does not model collate_fn changing the item type into a batch mapping.
    dataloader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        samples,  # type: ignore[arg-type]
        batch_size=2,
        shuffle=False,
        collate_fn=SpeechTextCollator(tokenizer, blank_token_id=0),
    )

    batch = next(iter(dataloader))

    torch.testing.assert_close(batch["waveforms"], torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.0]]))
    torch.testing.assert_close(batch["waveform_lengths"], torch.tensor([3, 2]))
    torch.testing.assert_close(batch["labels"], torch.tensor([[2, 3], [2, 0]]))
    torch.testing.assert_close(batch["label_lengths"], torch.tensor([2, 1]))
