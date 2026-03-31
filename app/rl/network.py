"""
Deep Q-Network architecture.

A simple 3-layer MLP sufficient for the 4-dimensional traffic state space.
Two instances are kept: online (trained every step) and target (frozen,
synced every `target_update_freq` episodes) for stable bootstrapping.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNNetwork(nn.Module):
    """
    MLP Q-network: maps state → Q-value for each action.

    Architecture
    ------------
    Input  : state_dim  (default 4: N/S/E/W queue lengths)
    Hidden : 128 → 128 → 64, ReLU activations, LayerNorm for stability
    Output : action_dim (default 2: NS-green / EW-green)
    """

    def __init__(self, state_dim: int = 4, action_dim: int = 2, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
