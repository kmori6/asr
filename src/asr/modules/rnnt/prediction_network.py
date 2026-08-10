import torch
import torch.nn as nn

type PredictionState = tuple[torch.Tensor, torch.Tensor]


class PredictionNetwork(nn.Module):
    """LSTM-based prediction network used by an RNN-T model.

    Proposed in A. Graves et al., "Speech recognition with deep recrrent neural networks,"
    in ICASSP, 2013, pp. 6645-6649.

    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        dropout_rate: float,
        blank_token_id: int,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate must satisfy 0 <= dropout_rate < 1")
        if not 0 <= blank_token_id < vocab_size:
            raise ValueError("blank_token_id must be a valid vocabulary index")

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.blank_token_id = blank_token_id
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=blank_token_id)
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        state: PredictionState | None = None,
    ) -> tuple[torch.Tensor, PredictionState]:
        """
        Args:
            tokens: Token IDs with shape ``(batch, sequence_length)``.
            state: Optional LSTM hidden and cell states, each with shape
                ``(num_layers, batch, hidden_size)``. ``None`` initializes both
                states to zero.

        Returns:
            Prediction embeddings with shape
            ``(batch, sequence_length, hidden_size)`` and the updated LSTM state.
        """
        if tokens.ndim != 2:
            raise ValueError(f"tokens must have shape (batch, sequence_length), but got {tuple(tokens.shape)}")
        if tokens.dtype != torch.long:
            raise ValueError("tokens must have dtype torch.long")
        if tokens.shape[1] == 0:
            raise ValueError("tokens must contain at least one token")

        x = self.dropout(self.embedding(tokens))
        if state is not None:
            expected_shape = (self.num_layers, tokens.shape[0], self.hidden_size)
            hidden_state, cell_state = state
            if hidden_state.shape != expected_shape or cell_state.shape != expected_shape:
                raise ValueError(f"hidden and cell states must have shape {expected_shape}")
            if hidden_state.device != x.device or cell_state.device != x.device:
                raise ValueError("tokens and states must be on the same device")

        x, next_state = self.lstm(x, state)
        return x, next_state
