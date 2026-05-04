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
            ft_size: int = 1024,
            l1_size: int = 256,
            l2_size: int = 64,
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
            counts = mask.sum(dim=1, dtype=torch.long)
            offsets = torch.empty(counts.numel() + 1, dtype=torch.long, device=indices.device)
            offsets[0] = 0
            offsets[1:] = torch.cumsum(counts, dim=0)
            flat_indices = clamped[mask]
            if flat_indices.numel() == 0:
                return self.ft_bias.unsqueeze(0).expand(indices.shape[0], -1)

            real_acc = torch.nn.functional.embedding_bag(
                flat_indices,
                self.ft.weight,
                offsets,
                mode="sum",
                include_last_offset=True,
            )
            factor_acc = torch.nn.functional.embedding_bag(
                torch.remainder(flat_indices, self.num_factor_features),
                self.ft_factor.weight,
                offsets,
                mode="sum",
                include_last_offset=True,
            )
            return real_acc + factor_acc + self.ft_bias

        def forward(
            self,
            white_indices: torch.Tensor,
            black_indices: torch.Tensor,
            stm: torch.Tensor,
        ) -> torch.Tensor:
            white_acc = self._accumulate(white_indices)
            black_acc = self._accumulate(black_indices)
            stm_bool = stm.ge(0.5)
            us = torch.where(stm_bool, white_acc, black_acc)
            them = torch.where(stm_bool, black_acc, white_acc)
            combined = torch.cat([us, them], dim=1)
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
