"""
DQN Agent
=========
Implements:
  • Epsilon-greedy action selection with linear decay
  • Online Q-network + frozen target network
  • Huber loss (smooth L1) for robust gradient updates
  • Hard target-network sync every `target_update_freq` episodes
  • Async-friendly training loop that yields EpisodeResult after each episode
    so FastAPI can stream progress over WebSocket
"""

from __future__ import annotations

import asyncio
import copy
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from app.rl.network import DQNNetwork
from app.rl.replay_buffer import ReplayBuffer
from app.rl.environment import TrafficSignalEnv, TrafficEnvConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass (matches POST /train request body)
# ---------------------------------------------------------------------------

@dataclass
class DQNConfig:
    # Training
    episodes:           int   = 500
    max_steps:          int   = 200
    batch_size:         int   = 64
    learning_rate:      float = 3e-4
    gamma:              float = 0.99      # discount factor

    # Replay buffer
    buffer_capacity:    int   = 10_000
    min_buffer_size:    int   = 500       # steps before training starts

    # Epsilon-greedy exploration
    epsilon_start:      float = 1.0
    epsilon_end:        float = 0.05
    epsilon_decay:      float = 0.995     # multiplied each episode

    # Target network
    target_update_freq: int   = 10        # sync every N episodes

    # Network
    hidden_size:        int   = 128

    # Environment
    env_config: TrafficEnvConfig = field(default_factory=TrafficEnvConfig)


# ---------------------------------------------------------------------------
# Per-episode result (streamed to frontend)
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    episode:        int
    total_episodes: int
    reward:         float
    avg_wait:       float
    throughput:     float
    epsilon:        float
    loss:           float | None
    duration_s:     float
    best_reward:    float


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    def __init__(self, config: DQNConfig, state_dim: int = 4, action_dim: int = 2):
        self.cfg        = config
        self.state_dim  = state_dim
        self.action_dim = action_dim

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("DQNAgent using device: %s", self.device)

        # Networks
        self.online_net = DQNNetwork(state_dim, action_dim, config.hidden_size).to(self.device)
        self.target_net = copy.deepcopy(self.online_net).to(self.device)
        self.target_net.eval()

        # Optimiser & loss
        self.optimiser = optim.Adam(self.online_net.parameters(), lr=config.learning_rate)
        self.loss_fn   = nn.SmoothL1Loss()   # Huber loss

        # Replay buffer
        self.buffer = ReplayBuffer(capacity=config.buffer_capacity)

        # Training state
        self.epsilon     = config.epsilon_start
        self.best_reward = float("-inf")
        self.episode     = 0
        self._stop       = False

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy: explore randomly or exploit Q-network."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_vals  = self.online_net(state_t)
            return int(q_vals.argmax(dim=1).item())

    def predict(self, state: np.ndarray) -> dict:
        """Greedy action + Q-values (for /predict endpoint)."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_vals  = self.online_net(state_t).squeeze(0)
            action  = int(q_vals.argmax().item())
        return {
            "action":       action,
            "action_name":  TrafficSignalEnv.ACTION_NAMES[action],
            "q_values":     q_vals.cpu().tolist(),
            "epsilon":      round(self.epsilon, 4),
        }

    # ------------------------------------------------------------------
    # Single gradient update
    # ------------------------------------------------------------------

    def _learn(self) -> float | None:
        if len(self.buffer) < self.cfg.min_buffer_size:
            return None

        try:
            states, actions, rewards, next_states, dones = self.buffer.sample(self.cfg.batch_size)
        except ValueError:
            return None

        states      = states.to(self.device)
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)

        # ── Current Q-values ────────────────────────────────────────
        # Q(s, a) for the action that was actually taken
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # ── Target Q-values (Bellman) ────────────────────────────────
        with torch.no_grad():
            # max Q(s', a') from frozen target network
            max_next_q = self.target_net(next_states).max(dim=1).values
            target_q   = rewards + self.cfg.gamma * max_next_q * (1.0 - dones)

        # ── Huber loss + gradient step ───────────────────────────────
        loss = self.loss_fn(current_q, target_q)
        self.optimiser.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimiser.step()

        return float(loss.item())

    # ------------------------------------------------------------------
    # Target network sync
    # ------------------------------------------------------------------

    def _sync_target(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ------------------------------------------------------------------
    # Full training loop (async generator → yields after each episode)
    # ------------------------------------------------------------------

    async def train(
        self,
        on_episode: Callable[[EpisodeResult], None] | None = None,
    ) -> AsyncGenerator[EpisodeResult, None]:
        """
        Async generator.  Caller does:

            async for result in agent.train():
                await ws.send_json(result.__dict__)
        """
        env      = TrafficSignalEnv(self.cfg.env_config)
        self._stop = False

        for ep in range(1, self.cfg.episodes + 1):
            if self._stop:
                break

            self.episode = ep
            t0           = time.perf_counter()

            state, _ = env.reset(seed=ep)
            ep_reward    = 0.0
            ep_wait_sum  = 0.0
            ep_thru_sum  = 0
            ep_loss_sum  = 0.0
            ep_loss_cnt  = 0
            last_info    = {}

            for step in range(self.cfg.max_steps):
                action            = self.select_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)

                self.buffer.push(state, action, reward, next_state, terminated or truncated)

                state        = next_state
                ep_reward   += reward
                ep_wait_sum += info.get("avg_wait", 0.0)
                ep_thru_sum += info.get("throughput", 0)
                last_info    = info

                loss = self._learn()
                if loss is not None:
                    ep_loss_sum += loss
                    ep_loss_cnt += 1

                if terminated or truncated:
                    break

            # ── Epsilon decay ────────────────────────────────────────
            self.epsilon = max(
                self.cfg.epsilon_end,
                self.epsilon * self.cfg.epsilon_decay,
            )

            # ── Target network sync ──────────────────────────────────
            if ep % self.cfg.target_update_freq == 0:
                self._sync_target()
                logger.debug("Episode %d: target network synced", ep)

            # ── Track best ──────────────────────────────────────────
            if ep_reward > self.best_reward:
                self.best_reward = ep_reward

            steps_done = last_info.get("step", self.cfg.max_steps)
            avg_wait   = ep_wait_sum / max(steps_done, 1)
            avg_thru   = ep_thru_sum / max(steps_done, 1)
            avg_loss   = ep_loss_sum / ep_loss_cnt if ep_loss_cnt > 0 else None

            result = EpisodeResult(
                episode        = ep,
                total_episodes = self.cfg.episodes,
                reward         = round(ep_reward, 3),
                avg_wait       = round(avg_wait, 3),
                throughput     = round(avg_thru, 3),
                epsilon        = round(self.epsilon, 4),
                loss           = round(avg_loss, 5) if avg_loss is not None else None,
                duration_s     = round(time.perf_counter() - t0, 3),
                best_reward    = round(self.best_reward, 3),
            )

            logger.info(
                "Ep %d/%d | R=%.1f | wait=%.2f | ε=%.3f | loss=%s",
                ep, self.cfg.episodes,
                ep_reward, avg_wait, self.epsilon,
                f"{avg_loss:.5f}" if avg_loss else "—",
            )

            if on_episode:
                on_episode(result)

            yield result

            # Yield control to event loop so WebSocket sends can proceed
            await asyncio.sleep(0)

    def stop(self) -> None:
        """Signal the training loop to halt after the current episode."""
        self._stop = True

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save({
            "online_net":  self.online_net.state_dict(),
            "target_net":  self.target_net.state_dict(),
            "optimiser":   self.optimiser.state_dict(),
            "epsilon":     self.epsilon,
            "episode":     self.episode,
            "best_reward": self.best_reward,
            "config":      self.cfg.__dict__,
        }, path)
        logger.info("Model saved to %s", path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimiser.load_state_dict(ckpt["optimiser"])
        self.epsilon     = ckpt.get("epsilon",     self.cfg.epsilon_end)
        self.episode     = ckpt.get("episode",     0)
        self.best_reward = ckpt.get("best_reward", float("-inf"))
        logger.info("Model loaded from %s (ep %d)", path, self.episode)
