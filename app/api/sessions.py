"""
Sessions API
============
GET  /sessions              — list all training sessions (paginated)
GET  /sessions/{id}         — single session metadata
GET  /sessions/{id}/episodes — episode logs for a session (for chart replay)
DELETE /sessions/{id}       — delete a session and its episode logs

Model checkpoints
=================
POST /sessions/{id}/save    — save current agent weights to disk
POST /sessions/{id}/load    — load weights from a previous checkpoint
GET  /sessions/{id}/checkpoint — check if a checkpoint file exists
"""

from __future__ import annotations

import os
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db, TrainingSession, EpisodeLog

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/sessions", tags=["sessions"])

CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ── Response schemas ──────────────────────────────────────────────────────────

class SessionSummary(BaseModel):
    id:             str
    status:         str
    algorithm:      str
    total_episodes: int
    best_reward:    float
    created_at:     str
    finished_at:    Optional[str] = None

    model_config = {"from_attributes": True}


class EpisodeRecord(BaseModel):
    episode:    int
    reward:     float
    avg_wait:   float
    throughput: float
    epsilon:    float
    loss:       Optional[float]
    duration_s: float

    model_config = {"from_attributes": True}


class SessionDetail(SessionSummary):
    config:        dict
    episode_count: int


class CheckpointStatus(BaseModel):
    session_id: str
    exists:     bool
    path:       Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _checkpoint_path(session_id: str) -> str:
    return os.path.join(CHECKPOINT_DIR, f"{session_id}.pt")


def _session_to_summary(s: TrainingSession) -> SessionSummary:
    return SessionSummary(
        id             = s.id,
        status         = s.status,
        algorithm      = s.algorithm,
        total_episodes = s.total_episodes,
        best_reward    = round(s.best_reward or 0.0, 3),
        created_at     = s.created_at.isoformat() if s.created_at else "",
        finished_at    = s.finished_at.isoformat() if s.finished_at else None,
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("", response_model=list[SessionSummary])
async def list_sessions(
    limit:  int = Query(20, ge=1, le=200),
    offset: int = Query(0,  ge=0),
    status: Optional[str] = Query(None, description="Filter by status: running|completed|stopped|error"),
    db:     AsyncSession = Depends(get_db),
) -> list[SessionSummary]:
    """Return all training sessions, newest first."""
    q = select(TrainingSession).order_by(TrainingSession.created_at.desc()).offset(offset).limit(limit)
    if status:
        q = q.where(TrainingSession.status == status)
    result = await db.execute(q)
    sessions = result.scalars().all()
    return [_session_to_summary(s) for s in sessions]


@router.get("/{session_id}", response_model=SessionDetail)
async def get_session(
    session_id: str,
    db:         AsyncSession = Depends(get_db),
) -> SessionDetail:
    result = await db.execute(select(TrainingSession).where(TrainingSession.id == session_id))
    s = result.scalar_one_or_none()
    if not s:
        raise HTTPException(404, f"Session {session_id!r} not found.")

    ep_count_result = await db.execute(
        select(func.count()).where(EpisodeLog.session_id == session_id)
    )
    ep_count = ep_count_result.scalar() or 0

    return SessionDetail(
        **_session_to_summary(s).model_dump(),
        config        = s.config or {},
        episode_count = ep_count,
    )


@router.get("/{session_id}/episodes", response_model=list[EpisodeRecord])
async def get_episodes(
    session_id: str,
    limit:      int = Query(500, ge=1, le=5000),
    offset:     int = Query(0,   ge=0),
    db:         AsyncSession = Depends(get_db),
) -> list[EpisodeRecord]:
    """Return episode logs for chart replay in the dashboard."""
    # Verify session exists
    exists = await db.execute(
        select(TrainingSession.id).where(TrainingSession.id == session_id)
    )
    if not exists.scalar_one_or_none():
        raise HTTPException(404, f"Session {session_id!r} not found.")

    q = (
        select(EpisodeLog)
        .where(EpisodeLog.session_id == session_id)
        .order_by(EpisodeLog.episode)
        .offset(offset)
        .limit(limit)
    )
    result = await db.execute(q)
    logs = result.scalars().all()

    return [
        EpisodeRecord(
            episode    = ep.episode,
            reward     = ep.reward,
            avg_wait   = ep.avg_wait,
            throughput = ep.throughput,
            epsilon    = ep.epsilon,
            loss       = ep.loss,
            duration_s = ep.duration_s,
        )
        for ep in logs
    ]


@router.delete("/{session_id}", status_code=200)
async def delete_session(
    session_id: str,
    db:         AsyncSession = Depends(get_db),
) -> dict:
    result = await db.execute(select(TrainingSession).where(TrainingSession.id == session_id))
    s = result.scalar_one_or_none()
    if not s:
        raise HTTPException(404, f"Session {session_id!r} not found.")

    await db.delete(s)   # cascade deletes episode_logs
    await db.commit()

    # Also remove checkpoint file if it exists
    ckpt = _checkpoint_path(session_id)
    if os.path.exists(ckpt):
        os.remove(ckpt)
        logger.info("Removed checkpoint: %s", ckpt)

    return {"deleted": session_id}


# ── Checkpoint endpoints ──────────────────────────────────────────────────────

@router.post("/{session_id}/save", response_model=CheckpointStatus)
async def save_checkpoint(
    session_id: str,
    db:         AsyncSession = Depends(get_db),
) -> CheckpointStatus:
    """Persist the current in-memory agent weights to disk."""
    from app.api.training import _agent, _session_id as active_id

    if _agent is None:
        raise HTTPException(404, "No agent in memory. Start training first.")
    if active_id != session_id:
        raise HTTPException(
            409,
            f"Active session is {active_id!r}, not {session_id!r}. "
            "Only the active agent can be saved.",
        )

    path = _checkpoint_path(session_id)
    _agent.save(path)

    # Update DB record
    await db.execute(
        TrainingSession.__table__.update()
        .where(TrainingSession.id == session_id)
        .values(best_reward=_agent.best_reward)
    )
    await db.commit()

    logger.info("Checkpoint saved: %s", path)
    return CheckpointStatus(session_id=session_id, exists=True, path=path)


@router.post("/{session_id}/load", response_model=CheckpointStatus)
async def load_checkpoint(
    session_id: str,
    db:         AsyncSession = Depends(get_db),
) -> CheckpointStatus:
    """Load weights from a saved checkpoint into the active agent."""
    from app.api.training import _agent, _session_id as active_id

    if _agent is None:
        raise HTTPException(404, "No agent in memory. Start training first.")
    if active_id != session_id:
        raise HTTPException(
            409,
            f"Active session is {active_id!r}. Load is only supported for the active agent.",
        )

    path = _checkpoint_path(session_id)
    if not os.path.exists(path):
        raise HTTPException(404, f"No checkpoint found at {path!r}.")

    _agent.load(path)
    return CheckpointStatus(session_id=session_id, exists=True, path=path)


@router.get("/{session_id}/checkpoint", response_model=CheckpointStatus)
async def checkpoint_status(session_id: str) -> CheckpointStatus:
    """Check whether a checkpoint file exists on disk for this session."""
    path = _checkpoint_path(session_id)
    exists = os.path.exists(path)
    return CheckpointStatus(
        session_id = session_id,
        exists     = exists,
        path       = path if exists else None,
    )
