"""
Unit tests — no database, no network, pure Python + torch.

Run with:
    pytest tests/test_dqn.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from app.rl.environment   import TrafficSignalEnv, TrafficEnvConfig
from app.rl.replay_buffer import ReplayBuffer
from app.rl.network       import DQNNetwork
from app.rl.agent         import DQNAgent, DQNConfig


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def env() -> TrafficSignalEnv:
    return TrafficSignalEnv(TrafficEnvConfig(max_steps=50))


@pytest.fixture
def agent() -> DQNAgent:
    cfg = DQNConfig(
        episodes        = 3,
        max_steps       = 20,
        batch_size      = 8,
        buffer_capacity = 200,
        min_buffer_size = 8,
        epsilon_decay   = 0.9,
    )
    return DQNAgent(cfg)


# ── Environment ───────────────────────────────────────────────────────────────

class TestTrafficSignalEnv:

    def test_reset_returns_correct_shape(self, env):
        obs, info = env.reset()
        assert obs.shape == (4,), "Observation must be (4,) — [N, S, E, W]"
        assert obs.dtype == np.float32

    def test_obs_normalised_in_0_1(self, env):
        obs, _ = env.reset()
        assert (obs >= 0.0).all() and (obs <= 1.0).all()

    def test_step_valid_actions(self, env):
        env.reset()
        for action in (0, 1):
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs.shape == (4,)
            assert isinstance(reward, float)
            assert reward <= 0, "Reward must be non-positive (negative waiting)"

    def test_invalid_action_raises(self, env):
        env.reset()
        with pytest.raises(AssertionError):
            env.step(2)

    def test_episode_terminates_within_max_steps(self, env):
        env.reset()
        for _ in range(env.cfg.max_steps + 10):
            _, _, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                break
        assert terminated or truncated, "Episode should have ended"

    def test_throughput_ns_clears_ns_queues(self):
        cfg = TrafficEnvConfig(
            arrival_rate_ns=0.0, arrival_rate_ew=0.0,
            throughput_ns=5, max_steps=10,
        )
        env = TrafficSignalEnv(cfg)
        obs, _ = env.reset()
        # Manually set queues: 10 vehicles in N & S, 0 in E & W
        env._queues = np.array([10, 10, 0, 0], dtype=np.float32)
        _, _, _, _, info = env.step(TrafficSignalEnv.ACTION_NS)
        # After NS green with throughput=5: N and S should each decrease by 5
        assert env._queues[0] == 5.0
        assert env._queues[1] == 5.0
        assert env._queues[2] == 0.0   # EW untouched

    def test_min_green_steps_prevents_rapid_switching(self):
        cfg = TrafficEnvConfig(min_green_steps=5, max_steps=20)
        env = TrafficSignalEnv(cfg)
        env.reset()
        # Force NS phase
        for _ in range(3):
            env.step(TrafficSignalEnv.ACTION_NS)
        phase_before = env._phase
        # Try switching to EW — should be blocked (< 5 steps held)
        env.step(TrafficSignalEnv.ACTION_EW)
        assert env._phase == phase_before, "Phase switch should be blocked before min_green_steps"

    def test_info_keys_present(self, env):
        env.reset()
        _, _, _, _, info = env.step(0)
        for key in ("step", "phase", "queues", "avg_wait", "throughput"):
            assert key in info, f"Missing info key: {key}"


# ── Replay Buffer ─────────────────────────────────────────────────────────────

class TestReplayBuffer:

    def test_push_and_len(self):
        buf = ReplayBuffer(capacity=100)
        assert len(buf) == 0
        buf.push(np.zeros(4), 0, -1.0, np.zeros(4), False)
        assert len(buf) == 1

    def test_capacity_respected(self):
        buf = ReplayBuffer(capacity=10)
        for i in range(20):
            buf.push(np.zeros(4), 0, float(-i), np.zeros(4), False)
        assert len(buf) == 10, "Buffer should not exceed capacity"

    def test_sample_shapes(self):
        buf = ReplayBuffer(capacity=100)
        for _ in range(32):
            buf.push(np.random.rand(4).astype(np.float32), 0, -1.0,
                     np.random.rand(4).astype(np.float32), False)
        s, a, r, ns, d = buf.sample(16)
        assert s.shape  == (16, 4)
        assert a.shape  == (16,)
        assert r.shape  == (16,)
        assert ns.shape == (16, 4)
        assert d.shape  == (16,)

    def test_sample_raises_when_too_small(self):
        buf = ReplayBuffer(capacity=100)
        buf.push(np.zeros(4), 0, 0.0, np.zeros(4), False)
        with pytest.raises(ValueError):
            buf.sample(10)

    def test_tensors_are_float(self):
        buf = ReplayBuffer(capacity=50)
        for _ in range(16):
            buf.push(np.ones(4, dtype=np.float32), 1, -2.0,
                     np.ones(4, dtype=np.float32), True)
        s, a, r, ns, d = buf.sample(8)
        assert s.dtype  == torch.float32
        assert a.dtype  == torch.int64
        assert d.dtype  == torch.float32


# ── DQN Network ───────────────────────────────────────────────────────────────

class TestDQNNetwork:

    def test_output_shape(self):
        net  = DQNNetwork(state_dim=4, action_dim=2)
        inp  = torch.zeros(1, 4)
        out  = net(inp)
        assert out.shape == (1, 2)

    def test_batch_output_shape(self):
        net  = DQNNetwork(state_dim=4, action_dim=2)
        inp  = torch.rand(32, 4)
        out  = net(inp)
        assert out.shape == (32, 2)

    def test_output_is_finite(self):
        net = DQNNetwork(state_dim=4, action_dim=2)
        out = net(torch.rand(8, 4))
        assert torch.isfinite(out).all()

    def test_gradients_flow(self):
        net  = DQNNetwork(state_dim=4, action_dim=2)
        inp  = torch.rand(4, 4, requires_grad=False)
        out  = net(inp).sum()
        out.backward()
        for name, param in net.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


# ── DQN Agent ─────────────────────────────────────────────────────────────────

class TestDQNAgent:

    def test_select_action_valid(self, agent):
        obs, _ = TrafficSignalEnv(TrafficEnvConfig()).reset()
        action  = agent.select_action(obs)
        assert action in (0, 1)

    def test_epsilon_greedy_explores(self, agent):
        """With epsilon=1 every action should be random — both actions seen in 50 calls."""
        agent.epsilon = 1.0
        obs, _ = TrafficSignalEnv(TrafficEnvConfig()).reset()
        actions = {agent.select_action(obs) for _ in range(50)}
        assert len(actions) == 2, "Should explore both actions with epsilon=1"

    def test_epsilon_greedy_exploits(self, agent):
        """With epsilon=0 the same greedy action is always selected."""
        agent.epsilon = 0.0
        obs, _ = TrafficSignalEnv(TrafficEnvConfig()).reset()
        actions = {agent.select_action(obs) for _ in range(20)}
        assert len(actions) == 1, "Should always exploit with epsilon=0"

    def test_learn_returns_none_before_buffer_ready(self, agent):
        # Buffer is empty — _learn should return None
        loss = agent._learn()
        assert loss is None

    def test_learn_returns_loss_after_fill(self, agent):
        env = TrafficSignalEnv(TrafficEnvConfig())
        obs, _ = env.reset()
        # Fill buffer past min_buffer_size
        for _ in range(agent.cfg.min_buffer_size + 10):
            action = agent.select_action(obs)
            next_obs, reward, term, trunc, _ = env.step(action)
            agent.buffer.push(obs, action, reward, next_obs, term or trunc)
            obs = next_obs if not (term or trunc) else env.reset()[0]

        loss = agent._learn()
        assert loss is not None
        assert loss >= 0.0

    def test_target_network_starts_equal_to_online(self, agent):
        for (n1, p1), (n2, p2) in zip(
            agent.online_net.named_parameters(),
            agent.target_net.named_parameters(),
        ):
            assert torch.allclose(p1, p2), f"Param {n1} differs between online and target"

    def test_sync_target_copies_weights(self, agent):
        # Train online net for a few steps so weights diverge
        env = TrafficSignalEnv(TrafficEnvConfig())
        obs, _ = env.reset()
        for _ in range(agent.cfg.min_buffer_size + 20):
            action = agent.select_action(obs)
            next_obs, reward, term, trunc, _ = env.step(action)
            agent.buffer.push(obs, action, reward, next_obs, term or trunc)
            obs = next_obs if not (term or trunc) else env.reset()[0]
            agent._learn()

        # Networks should have diverged
        any_diff = any(
            not torch.allclose(p1, p2)
            for p1, p2 in zip(agent.online_net.parameters(), agent.target_net.parameters())
        )
        assert any_diff, "Networks should have diverged after training steps"

        # After sync they should match again
        agent._sync_target()
        for p1, p2 in zip(agent.online_net.parameters(), agent.target_net.parameters()):
            assert torch.allclose(p1, p2), "Target should match online after sync"

    def test_predict_returns_valid_action(self, agent):
        obs, _ = TrafficSignalEnv(TrafficEnvConfig()).reset()
        result  = agent.predict(obs)
        assert result["action"] in (0, 1)
        assert result["action_name"] in ("NS green", "EW green")
        assert len(result["q_values"]) == 2

    @pytest.mark.asyncio
    async def test_training_loop_yields_results(self, agent):
        results = []
        async for r in agent.train():
            results.append(r)
        assert len(results) == agent.cfg.episodes
        assert all(r.reward <= 0 for r in results), "All rewards should be non-positive"
        # Epsilon should have decayed (decay=0.9 over 3 episodes: 1.0 → ~0.729)
        assert results[-1].epsilon < results[0].epsilon

    @pytest.mark.asyncio
    async def test_stop_halts_training(self):
        cfg   = DQNConfig(episodes=1000, max_steps=5, min_buffer_size=1, batch_size=2)
        agent = DQNAgent(cfg)
        count = 0
        async for _ in agent.train():
            count += 1
            if count == 3:
                agent.stop()
                break
        assert count == 3
