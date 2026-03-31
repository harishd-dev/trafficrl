"""
TrafficSignalEnv — a single-intersection traffic signal environment.

State  : np.ndarray shape (4,) — [north, south, east, west] queue lengths
         Each queue in [0, max_queue].
Actions: 0 = keep/set NS green  |  1 = keep/set EW green
Reward : −(total waiting vehicles) — agent minimises congestion.

Episode ends after `max_steps` steps or when all queues are empty.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Simulation parameters (all tunable via TrafficEnvConfig)
# ---------------------------------------------------------------------------

@dataclass
class TrafficEnvConfig:
    max_queue:          int   = 20     # max vehicles per lane
    max_steps:          int   = 200    # steps per episode
    arrival_rate_ns:    float = 0.3    # Poisson λ for N+S lanes (vehicles/step)
    arrival_rate_ew:    float = 0.3    # Poisson λ for E+W lanes
    throughput_ns:      int   = 3      # vehicles cleared per step when NS green
    throughput_ew:      int   = 3      # vehicles cleared per step when EW green
    min_green_steps:    int   = 3      # minimum steps before phase can switch
    yellow_penalty:     float = 2.0    # extra penalty per phase change


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class TrafficSignalEnv:
    """
    Lightweight gym-style environment (no gymnasium dependency required).
    Compatible with the gymnasium step API: returns (obs, reward, terminated, truncated, info).
    """

    ACTION_NS = 0
    ACTION_EW = 1
    ACTION_NAMES = {0: "NS green", 1: "EW green"}

    def __init__(self, config: TrafficEnvConfig | None = None):
        self.cfg = config or TrafficEnvConfig()

        # Observation / action spaces (shapes only — no gymnasium Spaces)
        self.observation_shape = (4,)
        self.n_actions = 2

        # Internal state
        self._queues      = np.zeros(4, dtype=np.float32)  # [N, S, E, W]
        self._step_count  = 0
        self._phase       = self.ACTION_NS   # current active phase
        self._phase_steps = 0                # steps held in current phase
        self._total_reward = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            np.random.seed(seed)

        # Random initial queues (0–30 % of max) to diversify training
        self._queues      = np.random.randint(0, self.cfg.max_queue // 3 + 1, size=4).astype(np.float32)
        self._step_count  = 0
        self._phase       = self.ACTION_NS
        self._phase_steps = 0
        self._total_reward = 0.0

        return self._get_obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert action in (0, 1), f"Invalid action {action}"

        self._step_count  += 1
        phase_changed      = False

        # ── Phase switching with minimum green enforcement ──────────
        if action != self._phase:
            if self._phase_steps >= self.cfg.min_green_steps:
                self._phase       = action
                self._phase_steps = 0
                phase_changed     = True
            # else: ignore the switch request (hold current phase)
        else:
            self._phase_steps += 1

        # ── Vehicle arrivals (Poisson) ──────────────────────────────
        arrivals = np.array([
            np.random.poisson(self.cfg.arrival_rate_ns),   # N
            np.random.poisson(self.cfg.arrival_rate_ns),   # S
            np.random.poisson(self.cfg.arrival_rate_ew),   # E
            np.random.poisson(self.cfg.arrival_rate_ew),   # W
        ], dtype=np.float32)
        self._queues = np.clip(self._queues + arrivals, 0, self.cfg.max_queue)

        # ── Throughput (vehicles cleared this step) ─────────────────
        if self._phase == self.ACTION_NS:
            # North & South move; East & West wait
            cleared_n = min(self._queues[0], self.cfg.throughput_ns)
            cleared_s = min(self._queues[1], self.cfg.throughput_ns)
            self._queues[0] -= cleared_n
            self._queues[1] -= cleared_s
        else:
            # East & West move; North & South wait
            cleared_e = min(self._queues[2], self.cfg.throughput_ew)
            cleared_w = min(self._queues[3], self.cfg.throughput_ew)
            self._queues[2] -= cleared_e
            self._queues[3] -= cleared_w

        # ── Reward ──────────────────────────────────────────────────
        waiting    = float(self._queues.sum())
        reward     = -waiting
        if phase_changed:
            reward -= self.cfg.yellow_penalty   # cost for phase switch

        self._total_reward += reward

        # ── Termination ─────────────────────────────────────────────
        terminated = bool(self._queues.sum() == 0)
        truncated  = self._step_count >= self.cfg.max_steps

        info = {
            "step":         self._step_count,
            "phase":        self.ACTION_NAMES[self._phase],
            "queues":       self._queues.tolist(),
            "total_reward": self._total_reward,
            "avg_wait":     waiting / 4.0,
            "throughput":   int((arrivals.sum()) - (self._queues.sum() - float(np.clip(self._queues - arrivals, 0, None).sum()))),
            "phase_changed": phase_changed,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def get_state(self) -> np.ndarray:
        """Return current observation (used for predict endpoint)."""
        return self._get_obs()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Normalise queues to [0, 1] for stable network input."""
        return (self._queues / self.cfg.max_queue).astype(np.float32)
