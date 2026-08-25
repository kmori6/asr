import torch
import torch.nn as nn


class JointNetwork(nn.Module):
    """Joint network that combines encoder and prediction representations.

    Proposed in A. Graves et al., "Speech recognition with deep recrrent neural networks,"
    in ICASSP, 2013, pp. 6645-6649.

    """

    def __init__(
        self,
        vocab_size: int,
        encoder_size: int,
        predictor_size: int,
        hidden_size: int,
        dropout_rate: float,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if encoder_size <= 0:
            raise ValueError("encoder_size must be positive")
        if predictor_size <= 0:
            raise ValueError("predictor_size must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate must satisfy 0 <= dropout_rate < 1")

        self.vocab_size = vocab_size
        self.encoder_size = encoder_size
        self.predictor_size = predictor_size
        self.encoder_projection = nn.Linear(encoder_size, hidden_size, bias=bias)
        self.predictor_projection = nn.Linear(predictor_size, hidden_size, bias=bias)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout_rate)
        self.output_projection = nn.Linear(hidden_size, vocab_size, bias=bias)

    def forward(self, encoder_outputs: torch.Tensor, predictor_outputs: torch.Tensor) -> torch.Tensor:
        """

        Args:
            encoder_outputs (torch.Tensor): Encoder representations with shape ``(batch, num_frames, encoder_size)``.
            predictor_outputs (torch.Tensor): Prediction representations with shape
                ``(batch, sequence_length, predictor_size)``.

        Returns:
            torch.Tensor: Logits with shape ``(batch, num_frames, sequence_length, vocab_size)``.
        """
        if encoder_outputs.ndim != 3:
            raise ValueError(
                "encoder_outputs must have shape (batch, num_frames, encoder_size), "
                f"but got {tuple(encoder_outputs.shape)}"
            )
        if predictor_outputs.ndim != 3:
            raise ValueError(
                "predictor_outputs must have shape (batch, sequence_length, predictor_size), "
                f"but got {tuple(predictor_outputs.shape)}"
            )
        if encoder_outputs.shape[0] != predictor_outputs.shape[0]:
            raise ValueError("encoder_outputs and predictor_outputs must have the same batch size")
        if encoder_outputs.shape[-1] != self.encoder_size:
            raise ValueError(f"expected encoder_size {self.encoder_size}, but got {encoder_outputs.shape[-1]}")
        if predictor_outputs.shape[-1] != self.predictor_size:
            raise ValueError(f"expected predictor_size {self.predictor_size}, but got {predictor_outputs.shape[-1]}")
        if encoder_outputs.device != predictor_outputs.device:
            raise ValueError("encoder_outputs and predictor_outputs must be on the same device")

        encoder_projection = self.encoder_projection(encoder_outputs).unsqueeze(2)
        predictor_projection = self.predictor_projection(predictor_outputs).unsqueeze(1)
        x = self.activation(encoder_projection + predictor_projection)
        x = self.dropout(x)
        return self.output_projection(x)
