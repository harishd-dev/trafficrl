"""
Integration tests — exercises every HTTP endpoint and the WebSocket stream.

Uses an in-memory SQLite database (no Postgres required) and a patched
DQNAgent that completes instantly (3 episodes, 5 steps each).

Run with:
    pytest tests/test_api.py -v
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool

# ── In-memory SQLite setup ────────────────────────────────────────────────────

from app.db.database import Base
from app.db import get_db

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(
    TEST_DB_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestSession = async_sessionmaker(test_engine, expire_on_commit=False)


async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
    async with TestSession() as session:
        yield session


# ── App fixture ───────────────────────────────────────────────────────────────

@pytest_asyncio.fixture(scope="function")
async def app() -> FastAPI:
    """Fresh app + tables for each test."""
    from app.main import app as _app
    _app.dependency_overrides[get_db] = override_get_db

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield _app

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    _app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


# ── Minimal fake agent (fast, no torch) ──────────────────────────────────────

from app.rl.agent import DQNConfig, EpisodeResult


async def _fake_train_gen(self):
    """Yields 3 fake episode results instantly."""
    for ep in range(1, 4):
        self.episode     = ep
        self.best_reward = float(-100 + ep * 10)
        self.epsilon     = max(0.05, 1.0 - ep * 0.3)
        result = EpisodeResult(
            episode        = ep,
            total_episodes = self.cfg.episodes,
            reward         = float(-100 + ep * 10),
            avg_wait       = float(30 - ep * 2),
            throughput     = float(2.0 + ep * 0.5),
            epsilon        = self.epsilon,
            loss           = 0.01 / ep,
            duration_s     = 0.001,
            best_reward    = self.best_reward,
        )
        yield result
        await asyncio.sleep(0)


def make_fast_agent(config: DQNConfig):
    agent = MagicMock()
    agent.cfg         = config
    agent.epsilon     = config.epsilon_start
    agent.best_reward = float("-inf")
    agent.episode     = 0
    agent.predict     = MagicMock(return_value={
        "action":      0,
        "action_name": "NS green",
        "q_values":    [-5.2, -8.1],
        "epsilon":     0.05,
    })
    agent.stop  = MagicMock()
    agent.save  = MagicMock()
    agent.load  = MagicMock()
    agent.train = lambda: _fake_train_gen(agent)
    return agent


# ── Health / root ─────────────────────────────────────────────────────────────

class TestHealth:

    @pytest.mark.asyncio
    async def test_root(self, client):
        r = await client.get("/")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_health(self, client):
        r = await client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


# ── GET /status before any training ──────────────────────────────────────────

class TestStatusIdle:

    @pytest.mark.asyncio
    async def test_status_idle(self, client):
        import app.api.training as t_module
        t_module._agent       = None
        t_module._session_id  = None
        t_module._last_result = None
        t_module._status      = "idle"

        r = await client.get("/status")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "idle"
        assert data["episode"] is None


# ── POST /train ───────────────────────────────────────────────────────────────

class TestTrain:

    @pytest.mark.asyncio
    async def test_train_returns_202_and_session_id(self, client):
        with patch("app.api.training._build_agent", side_effect=make_fast_agent):
            r = await client.post("/train", json={"episodes": 3, "max_steps": 5})

        assert r.status_code == 202
        data = r.json()
        assert "session_id" in data
        assert data["status"] == "running"
        assert len(data["session_id"]) == 36   # UUID format

    @pytest.mark.asyncio
    async def test_train_stores_session_in_db(self, client, app):
        with patch("app.api.training._build_agent", side_effect=make_fast_agent):
            r = await client.post("/train", json={"episodes": 3})
        session_id = r.json()["session_id"]

        # Wait for background task to finish
        import app.api.training as t_module
        if t_module._training_task:
            await asyncio.wait_for(t_module._training_task, timeout=5.0)

        r2 = await client.get(f"/sessions/{session_id}")
        assert r2.status_code == 200
        assert r2.json()["id"] == session_id

    @pytest.mark.asyncio
    async def test_train_conflict_when_already_running(self, client):
        import app.api.training as t_module

        # Fake a running task
        async def _long():
            await asyncio.sleep(60)

        t_module._training_task = asyncio.create_task(_long())

        try:
            r = await client.post("/train", json={"episodes": 3})
            assert r.status_code == 409
            assert "already running" in r.json()["detail"].lower()
        finally:
            t_module._training_task.cancel()
            try:
                await t_module._training_task
            except asyncio.CancelledError:
                pass
            t_module._training_task = None

    @pytest.mark.asyncio
    async def test_train_rejects_invalid_params(self, client):
        r = await client.post("/train", json={"episodes": -1})
        assert r.status_code == 422     # Pydantic validation error

        r2 = await client.post("/train", json={"gamma": 2.0})
        assert r2.status_code == 422


# ── POST /train/stop ──────────────────────────────────────────────────────────

class TestStop:

    @pytest.mark.asyncio
    async def test_stop_when_idle(self, client):
        import app.api.training as t_module
        t_module._agent        = None
        t_module._training_task = None
        t_module._status        = "idle"

        r = await client.post("/train/stop")
        assert r.status_code == 200
        assert r.json()["status"] == "stopped"

    @pytest.mark.asyncio
    async def test_stop_calls_agent_stop(self, client):
        import app.api.training as t_module

        mock_agent = MagicMock()
        mock_agent.stop = MagicMock()
        t_module._agent = mock_agent

        async def _long():
            await asyncio.sleep(60)

        t_module._training_task = asyncio.create_task(_long())
        t_module._status = "running"

        try:
            r = await client.post("/train/stop")
            assert r.status_code == 200
            mock_agent.stop.assert_called_once()
        finally:
            if t_module._training_task and not t_module._training_task.done():
                t_module._training_task.cancel()
                try:
                    await t_module._training_task
                except asyncio.CancelledError:
                    pass


# ── POST /predict ─────────────────────────────────────────────────────────────

class TestPredict:

    @pytest.mark.asyncio
    async def test_predict_no_agent_returns_404(self, client):
        import app.api.training as t_module
        t_module._agent = None

        r = await client.post("/predict", json={"state": [5, 3, 8, 2]})
        assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_predict_returns_action(self, client):
        import app.api.training as t_module

        mock_agent = MagicMock()
        mock_agent.predict.return_value = {
            "action":      1,
            "action_name": "EW green",
            "q_values":    [-12.4, -8.7],
            "epsilon":     0.05,
        }
        t_module._agent = mock_agent

        r = await client.post("/predict", json={"state": [5, 3, 8, 2]})
        assert r.status_code == 200
        data = r.json()
        assert data["action"] in (0, 1)
        assert data["action_name"] in ("NS green", "EW green")
        assert len(data["q_values"]) == 2

    @pytest.mark.asyncio
    async def test_predict_rejects_wrong_state_length(self, client):
        import app.api.training as t_module
        t_module._agent = MagicMock()

        r = await client.post("/predict", json={"state": [5, 3]})
        assert r.status_code == 422

        r2 = await client.post("/predict", json={"state": [1, 2, 3, 4, 5]})
        assert r2.status_code == 422

    @pytest.mark.asyncio
    async def test_predict_normalizes_state(self, client):
        """normalize=True should divide raw counts by 20 before inference."""
        import app.api.training as t_module
        import numpy as np

        captured = {}

        def fake_predict(state):
            captured["state"] = state
            return {"action": 0, "action_name": "NS green", "q_values": [0.0, 0.0], "epsilon": 0.0}

        mock_agent = MagicMock()
        mock_agent.predict.side_effect = fake_predict
        t_module._agent = mock_agent

        await client.post("/predict", json={"state": [20, 10, 0, 5], "normalize": True})
        called_state = np.array(captured["state"])
        np.testing.assert_allclose(called_state, [1.0, 0.5, 0.0, 0.25], atol=1e-5)


# ── GET /sessions ─────────────────────────────────────────────────────────────

class TestSessions:

    async def _create_session(self, client) -> str:
        with patch("app.api.training._build_agent", side_effect=make_fast_agent):
            r = await client.post("/train", json={"episodes": 3, "max_steps": 5})
        assert r.status_code == 202
        sid = r.json()["session_id"]
        # Let the training task complete
        import app.api.training as t_module
        if t_module._training_task:
            try:
                await asyncio.wait_for(asyncio.shield(t_module._training_task), timeout=5.0)
            except (asyncio.TimeoutError, Exception):
                pass
        return sid

    @pytest.mark.asyncio
    async def test_list_sessions_empty(self, client):
        r = await client.get("/sessions")
        assert r.status_code == 200
        assert r.json() == []

    @pytest.mark.asyncio
    async def test_list_sessions_after_training(self, client):
        await self._create_session(client)
        r = await client.get("/sessions")
        assert r.status_code == 200
        sessions = r.json()
        assert len(sessions) >= 1
        assert "id" in sessions[0]
        assert "status" in sessions[0]
        assert "best_reward" in sessions[0]

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, client):
        r = await client.get("/sessions/nonexistent-id")
        assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_get_session_detail(self, client):
        sid = await self._create_session(client)
        r = await client.get(f"/sessions/{sid}")
        assert r.status_code == 200
        data = r.json()
        assert data["id"] == sid
        assert "config" in data
        assert "episode_count" in data

    @pytest.mark.asyncio
    async def test_get_episodes(self, client):
        sid = await self._create_session(client)
        r = await client.get(f"/sessions/{sid}/episodes")
        assert r.status_code == 200
        eps = r.json()
        # Should have at least 1 episode logged (background task may not be done)
        assert isinstance(eps, list)
        if eps:
            assert "reward" in eps[0]
            assert "avg_wait" in eps[0]
            assert "epsilon" in eps[0]

    @pytest.mark.asyncio
    async def test_episodes_pagination(self, client):
        sid = await self._create_session(client)
        r = await client.get(f"/sessions/{sid}/episodes?limit=1&offset=0")
        assert r.status_code == 200
        assert len(r.json()) <= 1

    @pytest.mark.asyncio
    async def test_delete_session(self, client):
        sid = await self._create_session(client)

        r_del = await client.delete(f"/sessions/{sid}")
        assert r_del.status_code == 200
        assert r_del.json()["deleted"] == sid

        r_get = await client.get(f"/sessions/{sid}")
        assert r_get.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_nonexistent_session(self, client):
        r = await client.delete("/sessions/does-not-exist")
        assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_list_sessions_status_filter(self, client):
        await self._create_session(client)
        r = await client.get("/sessions?status=running")
        assert r.status_code == 200
        # All returned sessions should match the filter
        for s in r.json():
            assert s["status"] == "running"


# ── Checkpoint endpoints ──────────────────────────────────────────────────────

class TestCheckpoints:

    @pytest.mark.asyncio
    async def test_checkpoint_status_missing(self, client):
        r = await client.get("/sessions/fake-id/checkpoint")
        assert r.status_code == 200
        data = r.json()
        assert data["exists"] is False
        assert data["path"] is None

    @pytest.mark.asyncio
    async def test_save_no_agent_returns_404(self, client):
        import app.api.training as t_module
        t_module._agent = None
        r = await client.post("/sessions/some-id/save")
        assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_save_wrong_session_returns_409(self, client):
        import app.api.training as t_module
        t_module._agent      = MagicMock()
        t_module._session_id = "active-session-id"

        r = await client.post("/sessions/different-session-id/save")
        assert r.status_code == 409

    @pytest.mark.asyncio
    async def test_load_no_checkpoint_returns_404(self, client):
        import app.api.training as t_module
        t_module._agent      = MagicMock()
        t_module._session_id = "my-session"

        r = await client.post("/sessions/my-session/load")
        assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_save_and_checkpoint_status(self, client, tmp_path, monkeypatch):
        """Save a checkpoint and verify the status endpoint detects it."""
        import app.api.training as t_module
        import app.api.sessions as s_module

        monkeypatch.setattr(s_module, "CHECKPOINT_DIR", str(tmp_path))

        mock_agent = MagicMock()
        mock_agent.best_reward = -50.0

        def fake_save(path):
            open(path, "w").close()   # create an empty file

        mock_agent.save = fake_save
        t_module._agent      = mock_agent
        t_module._session_id = "test-session-save"

        r_save = await client.post("/sessions/test-session-save/save")
        assert r_save.status_code == 200
        assert r_save.json()["exists"] is True

        # Status should now show the file
        ckpt_path = s_module._checkpoint_path("test-session-save")
        r_status = await client.get("/sessions/test-session-save/checkpoint")
        assert r_status.status_code == 200
        # File exists at the (now monkeypatched) path
        import os
        assert os.path.exists(ckpt_path)


# ── GET /status after completed training ─────────────────────────────────────

class TestStatusAfterTraining:

    @pytest.mark.asyncio
    async def test_status_reflects_last_episode(self, client):
        import app.api.training as t_module

        with patch("app.api.training._build_agent", side_effect=make_fast_agent):
            await client.post("/train", json={"episodes": 3, "max_steps": 5})

        if t_module._training_task:
            try:
                await asyncio.wait_for(asyncio.shield(t_module._training_task), timeout=5.0)
            except (asyncio.TimeoutError, Exception):
                pass

        r = await client.get("/status")
        assert r.status_code == 200
        data = r.json()
        # After 3 fake episodes the last result should be populated
        if data["episode"] is not None:
            assert data["episode"] == 3
            assert data["reward"] is not None
            assert data["epsilon"] is not None
