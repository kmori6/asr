import torch
from pytest import approx

from asr.decoding import RNNTBeamSearch, RNNTBeamSearchResult
from asr.modules.rnnt import JointNetwork, PredictionNetwork
from asr.modules.rnnt.prediction_network import PredictionState


class _RulePredictionNetwork(PredictionNetwork):
    def __init__(self) -> None:
        super().__init__(
            vocab_size=3,
            hidden_size=1,
            num_layers=1,
            dropout_rate=0.0,
            blank_token_id=0,
        )
        self.consumed_tokens: list[int] = []

    def forward(
        self,
        tokens: torch.Tensor,
        state: PredictionState | None = None,
    ) -> tuple[torch.Tensor, PredictionState]:
        token_id = int(tokens.item())
        self.consumed_tokens.append(token_id)
        output = tokens.to(dtype=torch.float32).unsqueeze(-1)
        next_state = (
            torch.full((1, 1, 1), token_id, dtype=torch.float32, device=tokens.device),
            torch.full((1, 1, 1), token_id, dtype=torch.float32, device=tokens.device),
        )
        return output, next_state


class _RuleJointNetwork(JointNetwork):
    def __init__(self) -> None:
        super().__init__(
            vocab_size=3,
            encoder_size=1,
            predictor_size=1,
            hidden_size=1,
            dropout_rate=0.0,
        )

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        predictor_outputs: torch.Tensor,
    ) -> torch.Tensor:
        frame = int(encoder_outputs.item())
        previous_token = int(predictor_outputs.item())
        logits = torch.full((1, 1, 1, 3), -8.0, device=encoder_outputs.device)
        if (frame, previous_token) == (0, 0):
            logits[..., 1] = 4.0
            logits[..., 0] = 0.0
        elif (frame, previous_token) == (1, 1):
            logits[..., 2] = 4.0
            logits[..., 0] = 0.0
        else:
            logits[..., 0] = 4.0
        return logits


def test_rnnt_beam_search_continues_across_chunks_without_reconsuming_blank() -> None:
    prediction_network = _RulePredictionNetwork()
    searcher = RNNTBeamSearch(
        prediction_network=prediction_network,
        joint_network=_RuleJointNetwork(),
        beam_width=1,
        blank_token_id=0,
    )
    encoder_outputs = torch.tensor([[[0.0], [1.0]]])

    first_chunk_result = searcher.search(encoder_outputs[:, :1])
    chunked_result = searcher.search(encoder_outputs[:, 1:])

    full_searcher = RNNTBeamSearch(
        prediction_network=_RulePredictionNetwork(),
        joint_network=_RuleJointNetwork(),
        beam_width=1,
        blank_token_id=0,
    )
    full_result = full_searcher.search(encoder_outputs)

    assert isinstance(chunked_result, RNNTBeamSearchResult)
    assert first_chunk_result.token_ids == [1]
    assert chunked_result.token_ids == [1, 2]
    assert chunked_result.token_ids == full_result.token_ids
    assert chunked_result.score == approx(full_result.score)
    assert prediction_network.consumed_tokens == [0, 1, 2]
