"""
Training API
============
POST  /train          — start a DQN training session
POST  /train/stop     — gracefully stop the current session
GET   /status         — snapshot of current training state
WS    /ws/training    — real-time episode stream (JSON per episode)
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import (
    TrainRequest, TrainResponse,
    StatusResponse, EpisodeMessage,
)
from app.db import get_db, TrainingSession, EpisodeLog
from app.rl import DQNAgent, DQNConfig, EpisodeResult
from app.rl.environment import TrafficEnvConfig

logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# In-process training state (single-session; extend with a dict for multi)
# ---------------------------------------------------------------------------

_agent:          DQNAgent | None = None
_session_id:     str | None      = None
_training_task:  asyncio.Task | None = None
_ws_clients:     set[WebSocket] = set()
_last_result:    EpisodeResult | None = None
_status:         str = "idle"        # idle | running | completed | stopped | error


def _build_agent(req: TrainRequest) -> DQNAgent:
    env_cfg = TrafficEnvConfig(
        max_steps        = req.max_steps,
        arrival_rate_ns  = req.arrival_rate_ns,
        arrival_rate_ew  = req.arrival_rate_ew,
        throughput_ns    = req.throughput_ns,
        throughput_ew    = req.throughput_ew,
        min_green_steps  = req.min_green_steps,
    )
    dqn_cfg = DQNConfig(
        episodes           = req.episodes,
        max_steps          = req.max_steps,
        learning_rate      = req.learning_rate,
        gamma              = req.gamma,
        epsilon_start      = req.epsilon_start,
        epsilon_end        = req.epsilon_end,
        epsilon_decay      = req.epsilon_decay,
        batch_size         = req.batch_size,
        buffer_capacity    = req.buffer_capacity,
        target_update_freq = req.target_update_freq,
        hidden_size        = req.hidden_size,
        env_config         = env_cfg,
    )
    return DQNAgent(dqn_cfg)


async def _broadcast(payload: dict[str, Any]) -> None:
    """Send JSON to all connected WebSocket clients; drop dead connections."""
    dead: set[WebSocket] = set()
    for ws in _ws_clients:
        try:
            await ws.send_json(payload)
        except Exception:
            dead.add(ws)
    _ws_clients.difference_update(dead)


async def _run_training(
    agent:      DQNAgent,
    session_id: str,
    db_session: AsyncSession,
) -> None:
    global _status, _last_result

    try:
        async for result in agent.train():
            _last_result = result
            _status = "running"

            # Persist episode to DB
            ep_log = EpisodeLog(
                session_id = session_id,
                episode    = result.episode,
                reward     = result.reward,
                avg_wait   = result.avg_wait,
                throughput = result.throughput,
                epsilon    = result.epsilon,
                loss       = result.loss,
                duration_s = result.duration_s,
            )
            db_session.add(ep_log)

            # Flush every 10 episodes to avoid holding large transactions
            if result.episode % 10 == 0:
                await db_session.commit()

            # Broadcast to all WebSocket subscribers
            msg = EpisodeMessage(
                session_id     = session_id,
                episode        = result.episode,
                total_episodes = result.total_episodes,
                reward         = result.reward,
                avg_wait       = result.avg_wait,
                throughput     = result.throughput,
                epsilon        = result.epsilon,
                best_reward    = result.best_reward,
                loss           = result.loss,
                duration_s     = result.duration_s,
            )
            await _broadcast(msg.model_dump())

        # Training finished naturally
        _status = "completed"
        await db_session.execute(
            TrainingSession.__table__.update()
            .where(TrainingSession.id == session_id)
            .values(status="completed", finished_at=datetime.now(timezone.utc),
                    best_reward=agent.best_reward)
        )
        await db_session.commit()
        await _broadcast({"type": "done", "session_id": session_id, "status": "completed"})

    except asyncio.CancelledError:
        _status = "stopped"
        await db_session.execute(
            TrainingSession.__table__.update()
            .where(TrainingSession.id == session_id)
            .values(status="stopped", finished_at=datetime.now(timezone.utc))
        )
        await db_session.commit()
        await _broadcast({"type": "done", "session_id": session_id, "status": "stopped"})

    except Exception as exc:
        _status = "error"
        logger.exception("Training error: %s", exc)
        await _broadcast({"type": "error", "session_id": session_id, "message": str(exc)})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/train", response_model=TrainResponse, status_code=202)
async def start_training(
    req:    TrainRequest,
    db:     AsyncSession = Depends(get_db),
) -> TrainResponse:
    global _agent, _session_id, _training_task, _status

    # Allow only one concurrent session
    if _training_task and not _training_task.done():
        raise HTTPException(409, "A training session is already running. POST /train/stop first.")

    _agent      = _build_agent(req)
    _session_id = str(uuid.uuid4())
    _status     = "running"

    # Persist session metadata
    session = TrainingSession(
        id             = _session_id,
        status         = "running",
        algorithm      = req.algorithm,
        config         = req.model_dump(),
        total_episodes = req.episodes,
    )
    db.add(session)
    await db.commit()

    # Launch training as a background asyncio task
    _training_task = asyncio.create_task(
        _run_training(_agent, _session_id, db)
    )

    logger.info("Training started: session=%s episodes=%d", _session_id, req.episodes)
    return TrainResponse(
        session_id = _session_id,
        status     = "running",
        message    = f"DQN training started for {req.episodes} episodes.",
    )


@router.post("/train/stop", status_code=200)
async def stop_training() -> dict:
    global _agent, _training_task, _status

    if _agent:
        _agent.stop()   # sets _stop flag inside agent loop

    if _training_task and not _training_task.done():
        _training_task.cancel()
        try:
            await asyncio.wait_for(_training_task, timeout=5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

    _status = "stopped"
    return {"status": "stopped", "session_id": _session_id}


@router.get("/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    r = _last_result
    return StatusResponse(
        status         = _status,
        session_id     = _session_id,
        episode        = r.episode        if r else None,
        total_episodes = r.total_episodes if r else None,
        reward         = r.reward         if r else None,
        avg_wait       = r.avg_wait       if r else None,
        throughput     = r.throughput     if r else None,
        epsilon        = r.epsilon        if r else None,
        best_reward    = r.best_reward    if r else None,
        loss           = r.loss           if r else None,
    )


# ---------------------------------------------------------------------------
# WebSocket — real-time episode stream
# ---------------------------------------------------------------------------

@router.websocket("/ws/training")
async def ws_training(websocket: WebSocket) -> None:
    await websocket.accept()
    _ws_clients.add(websocket)
    logger.info("WS client connected (%d total)", len(_ws_clients))

    # Send current state immediately so the client can sync
    await websocket.send_json({
        "type":   "connected",
        "status": _status,
        "session_id": _session_id,
    })

    try:
        while True:
            # Keep connection alive; actual data is pushed via _broadcast
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients.discard(websocket)
        logger.info("WS client disconnected (%d remaining)", len(_ws_clients))
