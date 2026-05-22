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

    def crelu(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, 0.0, 1.0)


    class HalfKAv2HmNNUE(nn.Module):
        def __init__(
            self,
            *,
            num_features: int = 22_528,
            ft_size: int = 1024,
            hidden_size: int = 31,
            forward_size: int = 1,
            fc1_output_size: int = 32,
        ):
            super().__init__()
            if forward_size != 1:
                raise ValueError("HalfKAv2_hm uses exactly one forward lane")
            self.num_features = num_features
            self.ft_size = ft_size
            self.hidden_size = hidden_size
            self.forward_size = forward_size
            self.fc0_output_size = hidden_size + forward_size
            self.fc1_input_size = hidden_size * 2
            self.fc1_output_size = fc1_output_size

            self.ft = nn.Embedding(num_features, ft_size)
            self.ft_bias = nn.Parameter(torch.zeros(ft_size, dtype=torch.float32))
            self.fc0 = nn.Linear(ft_size * 2, self.fc0_output_size)
            self.fc1 = nn.Linear(self.fc1_input_size, fc1_output_size)
            self.fc2 = nn.Linear(fc1_output_size, 1)
            self.reset_parameters()

        def reset_parameters(self) -> None:
            nn.init.uniform_(self.ft.weight, -0.01, 0.01)
            nn.init.zeros_(self.ft_bias)
            nn.init.xavier_uniform_(self.fc0.weight)
            nn.init.zeros_(self.fc0.bias)
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.zeros_(self.fc1.bias)
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

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

            acc = torch.nn.functional.embedding_bag(
                flat_indices,
                self.ft.weight,
                offsets,
                mode="sum",
                include_last_offset=True,
            )
            return acc + self.ft_bias

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
            fc0_out = self.fc0(torch.cat([us, them], dim=1))
            hidden = fc0_out[:, : self.hidden_size]
            forward = fc0_out[:, self.hidden_size : self.hidden_size + self.forward_size]
            hidden_crelu = crelu(hidden)
            fc1_in = torch.cat([hidden_crelu.square(), hidden_crelu], dim=1)
            fc1_out = crelu(self.fc1(fc1_in))
            return self.fc2(fc1_out) + forward

else:

    class HalfKAv2HmNNUE:  # pragma: no cover - exercised only when torch is missing
        def __init__(self, *args, **kwargs):
            _require_torch()
