import torch

from asr.models import FastConformerTransformer
from asr.modules.conformer import FastConformer
from asr.modules.frontend import LogMelSpectrogram, SpecAugment
from asr.modules.transformer.cache import DecoderLayerCache


def _create_model() -> FastConformerTransformer:
    return FastConformerTransformer(
        frontend=LogMelSpectrogram(n_fft=16, hop_length=8, n_mels=4),
        spec_augment=SpecAugment(
            num_frequency_masks=0,
            max_frequency_mask_width=0,
            num_time_masks=0,
            max_time_mask_width=0,
        ),
        encoder=FastConformer(
            input_size=4,
            hidden_size=8,
            num_heads=2,
            kernel_size=5,
            num_blocks=1,
            dropout_rate=0.0,
            conv_channels=4,
        ),
        vocab_size=12,
        blank_token_id=0,
        ctc_loss_weight=0.2,
        decoder_hidden_size=8,
        decoder_num_layers=2,
        decoder_num_heads=2,
        decoder_feed_forward_size=16,
        decoder_dropout_rate=0.0,
        decoder_max_length=16,
        label_smoothing=0.0,
    )


def test_fast_conformer_transformer_computes_loss_and_gradients() -> None:
    torch.manual_seed(0)
    model = _create_model()
    waveforms = torch.randn(2, 256, requires_grad=True)

    metrics = model(
        waveforms=waveforms,
        waveform_lengths=torch.tensor([256, 200]),
        ctc_labels=torch.tensor([[4, 5], [6, 0]]),
        ctc_label_lengths=torch.tensor([2, 1]),
        decoder_input_ids=torch.tensor([[1, 4, 5], [1, 6, 0]]),
        decoder_attention_mask=torch.tensor([[1, 1, 1], [1, 1, 0]]),
        labels=torch.tensor([[4, 5, 2], [6, 2, -100]]),
    )

    assert metrics.keys() == {"loss", "ctc_loss", "cross_entropy_loss", "accuracy"}
    assert all(metric.shape == () and torch.isfinite(metric) for metric in metrics.values())
    torch.testing.assert_close(
        metrics["loss"],
        0.2 * metrics["ctc_loss"] + 0.8 * metrics["cross_entropy_loss"],
    )
    assert isinstance(model.ctc_loss_fn, torch.nn.CTCLoss)
    assert isinstance(model.cross_entropy_loss_fn, torch.nn.CrossEntropyLoss)
    assert model.output_projection.weight.data_ptr() == model.embedding.weight.data_ptr()
    metrics["loss"].backward()
    assert waveforms.grad is not None
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_fast_conformer_transformer_ignores_impossible_ctc_alignment() -> None:
    model = _create_model()
    metrics = model(
        waveforms=torch.randn(1, 128),
        waveform_lengths=torch.tensor([128]),
        ctc_labels=torch.tensor([[4, 5, 6]]),
        ctc_label_lengths=torch.tensor([3]),
        decoder_input_ids=torch.tensor([[1, 4, 5, 6]]),
        decoder_attention_mask=torch.ones(1, 4, dtype=torch.bool),
        labels=torch.tensor([[4, 5, 6, 2]]),
    )

    assert metrics["ctc_loss"].item() == 0.0
    assert torch.isfinite(metrics["loss"])
    assert metrics["cross_entropy_loss"].item() > 0.0


def test_fast_conformer_transformer_cached_decoder_matches_full_decoder() -> None:
    model = _create_model().eval()
    encoder_outputs = torch.randn(1, 4, 8)
    encoder_attention_mask = torch.tensor([[True, True, True, False]])
    token_ids = torch.tensor([[1, 4, 5, 2]])
    decoder_inputs = model.embed(token_ids)
    cross_attention_mask = encoder_attention_mask[:, None, :].expand(-1, token_ids.shape[1], -1)
    causal_mask = torch.ones(1, token_ids.shape[1], token_ids.shape[1], dtype=torch.bool).tril()
    expected = model.decoder(encoder_outputs, decoder_inputs, cross_attention_mask, causal_mask)

    caches: list[DecoderLayerCache] = []
    step_outputs = []
    for position in range(token_ids.shape[1]):
        step_input = model.embed(token_ids[:, position : position + 1], offset=position)
        step_output, caches = model.predict(
            encoder_outputs,
            step_input,
            encoder_attention_mask,
            caches,
        )
        step_outputs.append(step_output)

    torch.testing.assert_close(torch.cat(step_outputs, dim=1), expected)
    assert len(caches) == 2
    assert all(cache.self_attention.key.shape[2] == token_ids.shape[1] for cache in caches)
    assert all(cache.cross_attention.key.shape[2] == encoder_outputs.shape[1] for cache in caches)
