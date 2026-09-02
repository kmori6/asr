import torch
from transformers import Qwen3Config, Qwen3ForCausalLM, WavLMConfig, WavLMModel

from asr.models import WavLMQwen3


def _create_model() -> WavLMQwen3:
    speech_encoder = WavLMModel(
        WavLMConfig(
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            conv_dim=(8, 8),
            conv_stride=(2, 2),
            conv_kernel=(4, 2),
            conv_bias=False,
            num_conv_pos_embeddings=4,
            num_conv_pos_embedding_groups=2,
            mask_time_prob=0.0,
            hidden_dropout=0.0,
            activation_dropout=0.0,
            attention_dropout=0.0,
            feat_proj_dropout=0.0,
        )
    )
    language_model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            max_position_embeddings=64,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            attention_dropout=0.0,
        )
    )
    return WavLMQwen3(speech_encoder, language_model, audio_downsample_factor=2)


def test_wavlm_qwen3_computes_transcript_loss_and_gradients() -> None:
    torch.manual_seed(0)
    model = _create_model()
    waveforms = torch.randn(2, 80)
    output = model(
        waveforms=waveforms,
        waveform_lengths=torch.tensor([80, 64]),
        input_ids=torch.tensor([[1, 5, 6, 2], [1, 7, 2, 0]]),
        attention_mask=torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
        labels=torch.tensor([[-100, 5, 6, 2], [-100, 7, 2, -100]]),
    )

    assert output["logits"].shape == (2, 4, 32)
    assert output["loss"].shape == ()
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert model.audio_projector[1].weight.grad is not None
    assert model.speech_encoder.encoder.layers[0].attention.q_proj.weight.grad is not None
    assert model.language_model.model.layers[0].self_attn.q_proj.weight.grad is not None


def test_wavlm_qwen3_projects_audio_to_expected_rate_and_left_pads() -> None:
    model = _create_model().eval()
    embeddings, attention_mask = model.encode_audio(
        waveforms=torch.randn(2, 80),
        waveform_lengths=torch.tensor([80, 48]),
    )

    assert model.samples_per_audio_token == 8
    assert embeddings.shape == (2, 10, 8)
    assert attention_mask.sum(dim=1).tolist() == [10, 6]
    assert attention_mask[0].all()
    assert not attention_mask[1, :4].any()
    assert attention_mask[1, 4:].all()
    assert torch.count_nonzero(embeddings[1, :4]) == 0


def test_wavlm_qwen3_generates_only_new_tokens() -> None:
    model = _create_model().eval()
    generated_ids = model.generate(
        waveforms=torch.randn(1, 80),
        waveform_lengths=torch.tensor([80]),
        input_ids=torch.tensor([[1, 5]]),
        max_new_tokens=3,
    )

    assert generated_ids.shape[0] == 1
    assert 1 <= generated_ids.shape[1] <= 3
    assert torch.all((0 <= generated_ids) & (generated_ids < 32))


def test_wavlm_qwen3_freezes_only_speech_encoder() -> None:
    model = _create_model()
    model.freeze_speech_encoder()

    assert not any(parameter.requires_grad for parameter in model.speech_encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.audio_projector.parameters())
    assert all(parameter.requires_grad for parameter in model.language_model.parameters())


def test_wavlm_qwen3_freezes_only_wavlm_feature_encoder() -> None:
    model = _create_model()
    model.freeze_feature_encoder()

    assert not any(parameter.requires_grad for parameter in model.speech_encoder.feature_extractor.parameters())
    assert all(parameter.requires_grad for parameter in model.speech_encoder.feature_projection.parameters())
    assert all(parameter.requires_grad for parameter in model.speech_encoder.encoder.parameters())


def test_wavlm_qwen3_adds_language_model_lora() -> None:
    model = _create_model()
    model.add_language_model_lora(
        rank=2,
        alpha=4,
        dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    trainable_names = [name for name, parameter in model.language_model.named_parameters() if parameter.requires_grad]
    assert trainable_names
    assert all("lora_" in name for name in trainable_names)
