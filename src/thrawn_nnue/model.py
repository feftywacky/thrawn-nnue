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
        def __init__(self, num_features: int = 768, ft_size: int = 256, hidden_size: int = 32):
            super().__init__()
            self.num_features = num_features
            self.ft_size = ft_size
            self.hidden_size = hidden_size
            self.ft = nn.Embedding(num_features, ft_size)
            self.ft_bias = nn.Parameter(torch.zeros(ft_size, dtype=torch.float32))
            self.l1 = nn.Linear(ft_size * 2, hidden_size)
            self.output = nn.Linear(hidden_size, 1)
            self.reset_parameters()

        def reset_parameters(self) -> None:
            nn.init.uniform_(self.ft.weight, -0.05, 0.05)
            nn.init.zeros_(self.ft_bias)
            nn.init.xavier_uniform_(self.l1.weight)
            nn.init.zeros_(self.l1.bias)
            nn.init.xavier_uniform_(self.output.weight)
            nn.init.zeros_(self.output.bias)

        def _accumulate(self, indices: torch.Tensor) -> torch.Tensor:
            mask = indices.ge(0)
            clamped = indices.clamp_min(0)
            embeddings = self.ft(clamped)
            embeddings = embeddings * mask.unsqueeze(-1).to(dtype=embeddings.dtype)
            return embeddings.sum(dim=1) + self.ft_bias

        def forward(self, white_indices: torch.Tensor, black_indices: torch.Tensor, stm: torch.Tensor) -> torch.Tensor:
            white_acc = self._accumulate(white_indices)
            black_acc = self._accumulate(black_indices)
            stm_bool = stm.ge(0.5)
            combined = torch.where(
                stm_bool,
                torch.cat([white_acc, black_acc], dim=1),
                torch.cat([black_acc, white_acc], dim=1),
            )
            hidden = torch.clamp(combined, 0.0, 1.0)
            hidden = torch.clamp(self.l1(hidden), 0.0, 1.0)
            return self.output(hidden)

else:

    class DualPerspectiveA768NNUE:  # pragma: no cover - exercised only when torch is missing
        def __init__(self, *args, **kwargs):
            _require_torch()
