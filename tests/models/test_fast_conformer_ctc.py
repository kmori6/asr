import torch

from asr.models import FastConformerCTC, StreamingFastConformerCTC
from asr.modules.conformer import FastConformer, StreamingFastConformer
from asr.modules.frontend import LogMelSpectrogram, SpecAugment


def _frontend() -> LogMelSpectrogram:
    return LogMelSpectrogram(n_fft=16, hop_length=8, n_mels=4)


def _spec_augment() -> SpecAugment:
    return SpecAugment(0, 0, 0, 0)


def test_fast_conformer_ctc_computes_loss_with_full_context_encoder() -> None:
    torch.manual_seed(0)
    model = FastConformerCTC(
        frontend=_frontend(),
        spec_augment=_spec_augment(),
        encoder=FastConformer(
            input_size=4,
            hidden_size=8,
            num_heads=2,
            kernel_size=5,
            num_blocks=1,
            dropout_rate=0.0,
            conv_channels=4,
        ),
        vocab_size=8,
        blank_token_id=0,
    )
    waveforms = torch.randn(2, 256, requires_grad=True)
    waveform_lengths = torch.tensor([256, 200])
    labels = torch.tensor([[2, 3, 4], [5, 6, 0]])
    label_lengths = torch.tensor([3, 2])

    metrics = model(waveforms, waveform_lengths, labels, label_lengths)

    assert type(model.encoder) is FastConformer
    assert metrics.keys() == {"loss"}
    assert metrics["loss"].ndim == 0 and torch.isfinite(metrics["loss"])
    metrics["loss"].backward()
    assert waveforms.grad is not None
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_streaming_fast_conformer_ctc_inherits_and_matches_chunked_encoding() -> None:
    torch.manual_seed(0)
    model = StreamingFastConformerCTC(
        frontend=_frontend(),
        spec_augment=_spec_augment(),
        encoder=StreamingFastConformer(
            input_size=4,
            hidden_size=8,
            num_heads=2,
            kernel_size=5,
            num_blocks=1,
            dropout_rate=0.0,
            min_chunk_size=2,
            max_chunk_size=2,
            streaming_mask_probability=1.0,
            conv_channels=4,
        ),
        vocab_size=8,
        blank_token_id=0,
    )
    assert isinstance(model, FastConformerCTC)
    waveforms = torch.randn(2, 256, requires_grad=True)
    waveform_lengths = torch.tensor([256, 200])
    labels = torch.tensor([[2, 3, 4], [5, 6, 0]])
    label_lengths = torch.tensor([3, 2])

    metrics = model(waveforms, waveform_lengths, labels, label_lengths, chunk_size=2)

    assert metrics.keys() == {"loss"}
    assert metrics["loss"].ndim == 0 and torch.isfinite(metrics["loss"])
    metrics["loss"].backward()
    assert waveforms.grad is not None
    assert all(parameter.grad is not None for parameter in model.parameters())

    model.eval()
    full_outputs, full_lengths = model.encode(
        waveforms[:1].detach(),
        waveform_lengths[:1],
        chunk_size=2,
    )
    cache = None
    chunk_outputs = []
    chunks = waveforms[:1].detach().split((13, 47, 64, 132), dim=1)
    for index, chunk in enumerate(chunks):
        output, cache = model.encode_chunk(
            chunk,
            cache=cache,
            chunk_size=2,
            is_final=index == len(chunks) - 1,
        )
        chunk_outputs.append(output)

    streaming_outputs = torch.cat(chunk_outputs, dim=1)
    torch.testing.assert_close(streaming_outputs, full_outputs[:, : full_lengths[0]])
