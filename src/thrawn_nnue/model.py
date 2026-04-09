from __future__ import annotations

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None


def _require_torch():
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for model and training commands")


if torch is not None:

    class DualPerspectiveA768NNUE(nn.Module):
        def __init__(
            self,
            num_features: int = 768,
            ft_size: int = 256,
            hidden_size: int = 32,
            output_buckets: int = 1,
            head_type: str = "scalar",
        ):
            super().__init__()
            self.num_features = num_features
            self.ft_size = ft_size
            self.hidden_size = hidden_size
            self.output_buckets = output_buckets
            self.head_type = head_type
            self.ft = nn.Embedding(num_features, ft_size)
            self.ft_bias = nn.Parameter(torch.zeros(ft_size, dtype=torch.float32))
            self.l1 = nn.Linear(ft_size * 2, hidden_size)
            self.output = nn.Linear(hidden_size, output_buckets)
            self.wdl_output = (
                None
                if head_type == "scalar"
                else nn.Linear(hidden_size, output_buckets * 3)
            )
            self.reset_parameters()

        def reset_parameters(self) -> None:
            nn.init.uniform_(self.ft.weight, -0.05, 0.05)
            nn.init.zeros_(self.ft_bias)
            nn.init.xavier_uniform_(self.l1.weight)
            nn.init.zeros_(self.l1.bias)
            nn.init.xavier_uniform_(self.output.weight)
            nn.init.zeros_(self.output.bias)
            if self.wdl_output is not None:
                nn.init.xavier_uniform_(self.wdl_output.weight)
                nn.init.zeros_(self.wdl_output.bias)

        def _accumulate(self, indices: torch.Tensor) -> torch.Tensor:
            mask = indices.ge(0)
            clamped = indices.clamp_min(0)
            embeddings = self.ft(clamped)
            embeddings = embeddings * mask.unsqueeze(-1).to(dtype=embeddings.dtype)
            return embeddings.sum(dim=1) + self.ft_bias

        def forward(
            self,
            white_indices: torch.Tensor,
            black_indices: torch.Tensor,
            stm: torch.Tensor,
            output_bucket_indices: torch.Tensor | None = None,
            return_wdl: bool = False,
        ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
            white_acc = self._accumulate(white_indices)
            black_acc = self._accumulate(black_indices)
            stm_bool = stm.ge(0.5)
            combined = torch.where(
                stm_bool,
                torch.cat([white_acc, black_acc], dim=1),
                torch.cat([black_acc, white_acc], dim=1),
            )
            # Use SCReLU on the concatenated accumulators to avoid saturating
            # the first dense layer from sparse surviving activations.
            hidden = torch.square(torch.clamp(combined, 0.0, 1.0))
            hidden = torch.clamp(self.l1(hidden), 0.0, 1.0)
            outputs = self.output(hidden)
            wdl_outputs = None if self.wdl_output is None else self.wdl_output(hidden)
            if self.output_buckets == 1:
                if not return_wdl or wdl_outputs is None:
                    return outputs
                return outputs, wdl_outputs.reshape(-1, 3)
            if output_bucket_indices is None:
                raise ValueError("output_bucket_indices are required when output_buckets > 1")

            bucket_indices = output_bucket_indices.reshape(-1).to(device=outputs.device, dtype=torch.long)
            bucket_indices = bucket_indices.clamp_(0, self.output_buckets - 1)
            bucketed_value = outputs.gather(1, bucket_indices.unsqueeze(1))
            if not return_wdl or wdl_outputs is None:
                return bucketed_value
            wdl_outputs = wdl_outputs.reshape(-1, self.output_buckets, 3)
            bucketed_wdl = wdl_outputs[torch.arange(wdl_outputs.shape[0], device=wdl_outputs.device), bucket_indices]
            return bucketed_value, bucketed_wdl

else:

    class DualPerspectiveA768NNUE:  # pragma: no cover - exercised only when torch is missing
        def __init__(self, *args, **kwargs):
            _require_torch()
