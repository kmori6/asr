from collections.abc import Sequence

import torch
import torch.nn as nn

from asr.modules.transformer.cache import DecoderLayerCache


class Decoder(nn.Module):
    def __init__(self, layers: list[nn.Module], final_norm: nn.Module) -> None:
        super().__init__()
        if not layers:
            raise ValueError("layers must not be empty.")

        self.layers = nn.ModuleList(layers)
        self.final_norm = final_norm

    def forward(
        self,
        x_enc: torch.Tensor,
        x_dec: torch.Tensor,
        mask_enc: torch.Tensor,
        mask_dec: torch.Tensor,
    ) -> torch.Tensor:
        """

        Args:
            x_enc (torch.Tensor): Encoder output tensor (batch_size, source_sequence_length, d_model).
            x_dec (torch.Tensor): Decoder input tensor (batch_size, target_sequence_length, d_model).
            mask_enc (torch.Tensor): Encoder-decoder attention mask
                (batch_size, target_sequence_length, source_sequence_length).
            mask_dec (torch.Tensor): Decoder self-attention mask
                (batch_size, target_sequence_length, target_sequence_length).

        Returns:
            torch.Tensor: Output tensor (batch_size, target_sequence_length, d_model).
        """
        for layer in self.layers:
            x_dec = layer(x_enc, x_dec, mask_enc, mask_dec)

        return self.final_norm(x_dec)

    @torch.inference_mode()
    def predict(
        self,
        x_enc: torch.Tensor,
        x_dec: torch.Tensor,
        attention_mask: torch.Tensor,
        caches: list[DecoderLayerCache],
    ) -> tuple[torch.Tensor, list[DecoderLayerCache]]:
        """Forward pass with KV-cache for autoregressive inference.

        Args:
            x_enc (torch.Tensor): Encoder output tensor
                (batch_size, source_sequence_length, d_model).
            x_dec (torch.Tensor): Current decoder input tensor (batch_size, 1, d_model).
            attention_mask (torch.Tensor): Source attention mask
                (batch_size, source_sequence_length).
            caches (list[DecoderLayerCache]): Self- and cross-attention caches for each decoder
                layer.

        Returns:
            tuple[torch.Tensor, list[DecoderLayerCache]]: Output tensor and updated caches.
        """
        if caches and len(caches) != len(self.layers):
            raise ValueError(f"caches must be empty or contain {len(self.layers)} elements, but got {len(caches)}.")

        batch_size, target_length = x_dec.shape[:2]
        source_length = x_enc.shape[1]
        if target_length != 1:
            raise ValueError(f"x_dec must contain exactly one token, but got target length {target_length}.")
        if attention_mask.shape != (batch_size, source_length):
            raise ValueError(
                "attention_mask must have shape "
                f"({batch_size}, {source_length}), but got {tuple(attention_mask.shape)}."
            )

        cache_length = caches[0].self_attention.key.shape[2] if caches else 0

        mask_enc = attention_mask.to(device=x_dec.device, dtype=torch.bool).unsqueeze(1).expand(-1, target_length, -1)
        mask_dec = torch.ones(
            batch_size,
            target_length,
            cache_length + target_length,
            dtype=torch.bool,
            device=x_dec.device,
        )

        caches_per_layer: Sequence[DecoderLayerCache | None] = caches if caches else [None] * len(self.layers)
        new_caches = []
        for layer, cache in zip(self.layers, caches_per_layer):
            x_dec, new_cache = layer.predict(x_enc, x_dec, mask_enc, mask_dec, cache)  # type: ignore[operator]
            new_caches.append(new_cache)
        return self.final_norm(x_dec), new_caches
