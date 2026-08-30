import torch

from asr.models import TransformerLM


def _create_model() -> TransformerLM:
    return TransformerLM(
        vocab_size=16,
        hidden_size=8,
        num_heads=2,
        num_layers=2,
        feed_forward_size=16,
        dropout_rate=0.0,
        max_length=4,
    )


def test_transformer_lm_computes_causal_loss_and_gradients() -> None:
    model = _create_model()
    input_ids = torch.tensor([[2, 4, 5, 3], [2, 6, 3, 0]])
    attention_mask = torch.tensor([[True, True, True, True], [True, True, True, False]])
    labels = input_ids.masked_fill(~attention_mask, -100)

    output = model(input_ids, attention_mask, labels)

    assert output["logits"].shape == (2, 4, 16)
    assert output["loss"].ndim == 0
    assert torch.isfinite(output["loss"])

    output["loss"].backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_transformer_lm_cached_prediction_matches_full_forward() -> None:
    model = _create_model().eval()
    input_ids = torch.tensor([[2, 4, 5, 3], [2, 6, 7, 3]])

    expected = model(input_ids)["logits"]
    cache = None
    step_logits = []
    for position in range(input_ids.shape[1]):
        logits, cache = model.predict(input_ids[:, position : position + 1], cache)
        step_logits.append(logits)

    torch.testing.assert_close(torch.cat(step_logits, dim=1), expected)
    assert cache is not None
    assert len(cache.layers) == 2
    assert all(layer.key.shape == (2, 2, 4, 4) for layer in cache.layers)
