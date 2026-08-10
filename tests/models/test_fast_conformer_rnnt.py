import torch

from asr.models import FastConformerRNNT
from asr.modules.conformer import FastConformerEncoder
from asr.modules.frontend import LogMelSpectrogram, SpecAugment
from asr.modules.rnnt import JointNetwork, PredictionNetwork


def test_fast_conformer_rnnt_computes_rnnt_and_ctc_losses() -> None:
    torch.manual_seed(0)
    model = FastConformerRNNT(
        frontend=LogMelSpectrogram(n_fft=16, hop_length=8, n_mels=4),
        spec_augment=SpecAugment(
            num_frequency_masks=0,
            max_frequency_mask_width=0,
            num_time_masks=0,
            max_time_mask_width=0,
        ),
        encoder=FastConformerEncoder(
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
        prediction_network=PredictionNetwork(
            vocab_size=8,
            hidden_size=6,
            num_layers=1,
            dropout_rate=0.0,
            blank_token_id=0,
        ),
        joint_network=JointNetwork(
            vocab_size=8,
            encoder_size=8,
            predictor_size=6,
            hidden_size=10,
            dropout_rate=0.0,
        ),
        ctc_loss_weight=0.3,
        fastemit_lambda=0.004,
    )
    waveforms = torch.randn(2, 256, requires_grad=True)
    waveform_lengths = torch.tensor([256, 200])
    labels = torch.tensor([[2, 3, 4], [5, 6, 0]])
    label_lengths = torch.tensor([3, 2])

    metrics = model(waveforms, waveform_lengths, labels, label_lengths, chunk_size=2)

    assert metrics.keys() == {"loss", "rnnt_loss", "ctc_loss"}
    assert model.rnnt_loss_fn.fastemit_lambda == 0.004
    assert all(metric.ndim == 0 and torch.isfinite(metric) for metric in metrics.values())
    torch.testing.assert_close(metrics["loss"], metrics["rnnt_loss"] + 0.3 * metrics["ctc_loss"])

    metrics["loss"].backward()
    assert waveforms.grad is not None
    assert all(parameter.grad is not None for parameter in model.parameters())
