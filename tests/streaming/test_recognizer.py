import torch
from pytest import approx

from asr.decoding import CTCBeamSearch, RNNTBeamSearch
from asr.models import StreamingFastConformerCTC, StreamingFastConformerRNNT
from asr.modules.conformer import StreamingFastConformer
from asr.modules.frontend import LogMelSpectrogram, SpecAugment
from asr.modules.rnnt import JointNetwork, PredictionNetwork
from asr.streaming import AudioChunker, StreamingCTCRecognizer, StreamingRNNTRecognizer


def test_streaming_recognizer_matches_complete_sequence_decoding() -> None:
    torch.manual_seed(0)
    prediction_network = PredictionNetwork(
        vocab_size=3,
        hidden_size=4,
        num_layers=1,
        dropout_rate=0.0,
        blank_token_id=0,
    )
    joint_network = JointNetwork(
        vocab_size=3,
        encoder_size=8,
        predictor_size=4,
        hidden_size=5,
        dropout_rate=0.0,
    )
    model = StreamingFastConformerRNNT(
        frontend=LogMelSpectrogram(sample_rate=16, n_fft=8, hop_length=4, n_mels=4),
        spec_augment=SpecAugment(
            num_frequency_masks=0,
            max_frequency_mask_width=0,
            num_time_masks=0,
            max_time_mask_width=0,
        ),
        encoder=StreamingFastConformer(
            input_size=4,
            hidden_size=8,
            num_heads=2,
            kernel_size=5,
            num_blocks=1,
            dropout_rate=0.0,
            min_chunk_size=2,
            max_chunk_size=2,
            conv_channels=4,
        ),
        prediction_network=prediction_network,
        joint_network=joint_network,
        ctc_loss_weight=0.3,
        fastemit_lambda=0.004,
    ).eval()
    with torch.no_grad():
        joint_network.output_projection.weight.zero_()
        joint_network.output_projection.bias.copy_(torch.tensor([5.0, -5.0, -5.0]))

    waveform = torch.randn(64)
    encoder_outputs, encoder_lengths = model.encode(
        waveform.unsqueeze(0),
        torch.tensor([waveform.shape[0]]),
        chunk_size=2,
    )
    offline_searcher = RNNTBeamSearch(prediction_network, joint_network, beam_width=1, blank_token_id=0)
    offline_result = offline_searcher.search(encoder_outputs[:, : encoder_lengths[0]])

    streaming_searcher = RNNTBeamSearch(prediction_network, joint_network, beam_width=1, blank_token_id=0)
    recognizer = StreamingRNNTRecognizer(model, streaming_searcher, chunk_size=2)
    streaming_result = recognizer.recognize(
        waveform,
        AudioChunker(chunk_duration_ms=250, sample_rate=16),
    )

    assert streaming_result.token_ids == offline_result.token_ids
    assert streaming_result.score == approx(offline_result.score)


def test_streaming_ctc_recognizer_matches_complete_sequence_decoding() -> None:
    torch.manual_seed(0)
    model = StreamingFastConformerCTC(
        frontend=LogMelSpectrogram(sample_rate=16, n_fft=8, hop_length=4, n_mels=4),
        spec_augment=SpecAugment(0, 0, 0, 0),
        encoder=StreamingFastConformer(
            input_size=4,
            hidden_size=8,
            num_heads=2,
            kernel_size=5,
            num_blocks=1,
            dropout_rate=0.0,
            min_chunk_size=2,
            max_chunk_size=2,
            conv_channels=4,
        ),
        vocab_size=3,
        blank_token_id=0,
    ).eval()
    with torch.no_grad():
        model.output_projection.weight.zero_()
        model.output_projection.bias.copy_(torch.tensor([0.0, 5.0, -5.0]))

    waveform = torch.randn(64)
    encoder_outputs, encoder_lengths = model.encode(
        waveform.unsqueeze(0),
        torch.tensor([waveform.shape[0]]),
        chunk_size=2,
    )
    offline_result = CTCBeamSearch(beam_width=3, blank_token_id=0).search(
        model.logits(encoder_outputs[:, : encoder_lengths[0]])[0]
    )

    recognizer = StreamingCTCRecognizer(
        model=model,
        searcher=CTCBeamSearch(beam_width=3, blank_token_id=0),
        chunk_size=2,
    )
    streaming_result = recognizer.recognize(
        waveform,
        AudioChunker(chunk_duration_ms=250, sample_rate=16),
    )

    assert streaming_result.token_ids == offline_result.token_ids
    assert streaming_result.score == approx(offline_result.score)
