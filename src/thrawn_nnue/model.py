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

    class HalfKPNNUE(nn.Module):
        def __init__(
            self,
            *,
            num_features: int = 40960,
            num_factor_features: int = 640,
            ft_size: int = 256,
            l1_size: int = 32,
            l2_size: int = 32,
        ):
            super().__init__()
            self.num_features = num_features
            self.num_factor_features = num_factor_features
            self.ft_size = ft_size
            self.l1_size = l1_size
            self.l2_size = l2_size

            self.ft = nn.Embedding(num_features, ft_size)
            self.ft_factor = nn.Embedding(num_factor_features, ft_size)
            self.ft_bias = nn.Parameter(torch.zeros(ft_size, dtype=torch.float32))
            self.l1 = nn.Linear(ft_size * 2, l1_size)
            self.l2 = nn.Linear(l1_size, l2_size)
            self.output = nn.Linear(l2_size, 1)
            self.reset_parameters()

        def reset_parameters(self) -> None:
            nn.init.uniform_(self.ft.weight, -0.01, 0.01)
            nn.init.uniform_(self.ft_factor.weight, -0.01, 0.01)
            nn.init.zeros_(self.ft_bias)
            nn.init.xavier_uniform_(self.l1.weight)
            nn.init.zeros_(self.l1.bias)
            nn.init.xavier_uniform_(self.l2.weight)
            nn.init.zeros_(self.l2.bias)
            nn.init.xavier_uniform_(self.output.weight)
            nn.init.zeros_(self.output.bias)

        def _accumulate(self, indices: torch.Tensor) -> torch.Tensor:
            mask = indices.ge(0)
            clamped = indices.clamp_min(0)
            factor_indices = torch.remainder(clamped, self.num_factor_features)

            real_embeddings = self.ft(clamped)
            factor_embeddings = self.ft_factor(factor_indices)
            combined_embeddings = real_embeddings + factor_embeddings
            combined_embeddings = combined_embeddings * mask.unsqueeze(-1).to(dtype=combined_embeddings.dtype)
            return combined_embeddings.sum(dim=1) + self.ft_bias

        def forward(
            self,
            white_indices: torch.Tensor,
            black_indices: torch.Tensor,
            stm: torch.Tensor,
        ) -> torch.Tensor:
            white_acc = self._accumulate(white_indices)
            black_acc = self._accumulate(black_indices)
            stm_bool = stm.ge(0.5)
            combined = torch.where(
                stm_bool,
                torch.cat([white_acc, black_acc], dim=1),
                torch.cat([black_acc, white_acc], dim=1),
            )
            hidden0 = torch.clamp(combined, 0.0, 1.0)
            hidden1 = torch.clamp(self.l1(hidden0), 0.0, 1.0)
            hidden2 = torch.clamp(self.l2(hidden1), 0.0, 1.0)
            return self.output(hidden2)

        def coalesced_feature_transform(self) -> tuple[torch.Tensor, torch.Tensor]:
            repeats = self.num_features // self.num_factor_features
            factor_rows = self.ft_factor.weight.repeat(repeats, 1)
            return self.ft.weight + factor_rows, self.ft_bias

else:

    class HalfKPNNUE:  # pragma: no cover - exercised only when torch is missing
        def __init__(self, *args, **kwargs):
            _require_torch()
