"""
Experience Replay Buffer for DQN.

Stores (state, action, reward, next_state, done) tuples and
returns random mini-batches for training.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Transition:
    state:      np.ndarray
    action:     int
    reward:     float
    next_state: np.ndarray
    done:       bool


class ReplayBuffer:
    """Circular replay buffer with uniform random sampling."""

    def __init__(self, capacity: int = 10_000):
        self._buf: deque[Transition] = deque(maxlen=capacity)

    # ------------------------------------------------------------------
    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        self._buf.append(Transition(state, action, reward, next_state, done))

    # ------------------------------------------------------------------
    def sample(self, batch_size: int) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        if len(self._buf) < batch_size:
            raise ValueError(
                f"Buffer has {len(self._buf)} transitions, need {batch_size}"
            )

        batch = random.sample(self._buf, batch_size)

        states      = torch.FloatTensor(np.array([t.state      for t in batch]))
        actions     = torch.LongTensor( np.array([t.action     for t in batch]))
        rewards     = torch.FloatTensor(np.array([t.reward     for t in batch]))
        next_states = torch.FloatTensor(np.array([t.next_state for t in batch]))
        dones       = torch.FloatTensor(np.array([t.done       for t in batch], dtype=np.float32))

        return states, actions, rewards, next_states, dones

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._buf)

    @property
    def is_ready(self) -> bool:
        return len(self._buf) > 0
