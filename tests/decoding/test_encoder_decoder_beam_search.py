import torch
from pytest import approx

from asr.decoding import EncoderDecoderBeamSearch, EncoderDecoderBeamSearchResult
from asr.modules.transformer.cache import DecoderLayerCache, KVCache


class _FakeModel:
    def __init__(self) -> None:
        self.position = 0
        self.batch_sizes: list[int] = []
        self.cache_batch_sizes: list[int] = []

    def embed(self, token_ids: torch.Tensor, offset: int = 0) -> torch.Tensor:
        self.position = offset
        return token_ids.float().unsqueeze(-1)

    def predict(
        self,
        encoder_outputs: torch.Tensor,
        decoder_inputs: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        caches: list[DecoderLayerCache],
    ) -> tuple[torch.Tensor, list[DecoderLayerCache]]:
        batch_size = decoder_inputs.shape[0]
        self.batch_sizes.append(batch_size)
        assert encoder_outputs.shape[0] == batch_size
        assert encoder_attention_mask.shape[0] == batch_size

        new_value = decoder_inputs[:, None, :, :]
        if caches:
            self.cache_batch_sizes.append(caches[0].self_attention.key.shape[0])
            self_key = torch.cat((caches[0].self_attention.key, new_value), dim=2)
            cross_attention = caches[0].cross_attention
        else:
            self_key = new_value
            cross_attention = KVCache(key=new_value, value=new_value)
        cache = DecoderLayerCache(
            self_attention=KVCache(key=self_key, value=self_key),
            cross_attention=cross_attention,
        )
        return decoder_inputs, [cache]

    def logits(self, decoder_outputs: torch.Tensor) -> torch.Tensor:
        batch_size = decoder_outputs.shape[0]
        logits = torch.full((batch_size, 1, 4), -20.0, device=decoder_outputs.device)
        logits[:, :, 0] = 20.0  # BOS must not be generated after the initial position.
        if self.position == 0:
            logits[:, :, 1] = 0.0
            logits[:, :, 2] = -0.1
        elif self.position == 1:
            previous_token_ids = decoder_outputs[:, 0, 0].long()
            logits[torch.arange(batch_size), 0, previous_token_ids] = 0.0
        else:
            logits[:, :, 3] = 0.0
        return logits


def test_encoder_decoder_beam_search_batches_beams_and_reorders_caches() -> None:
    model = _FakeModel()
    searcher = EncoderDecoderBeamSearch(model, bos_token_id=0, eos_token_id=3)

    result = searcher.search(
        encoder_outputs=torch.zeros(1, 2, 4),
        encoder_attention_mask=torch.tensor([[True, False]]),
        beam_size=2,
        max_new_tokens=4,
        length_penalty=0.6,
    )

    assert isinstance(result, EncoderDecoderBeamSearchResult)
    assert result.token_ids[0] == 0
    assert result.token_ids[-1] == 3
    assert model.batch_sizes == [1, 2, 2]
    assert model.cache_batch_sizes == [2, 2]


def test_encoder_decoder_beam_search_uses_gnmt_length_penalty() -> None:
    assert EncoderDecoderBeamSearch._length_penalty(1, 0.6) == approx(1.0)
    assert EncoderDecoderBeamSearch._length_penalty(7, 0.6) == approx(2.0**0.6)
