"""
ORM models for persisting training sessions and per-episode metrics.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    String, Float, Integer, Boolean, DateTime, JSON,
    ForeignKey, func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class TrainingSession(Base):
    """One row per POST /train call."""

    __tablename__ = "training_sessions"

    id:            Mapped[str]      = mapped_column(String(36), primary_key=True, default=_uuid)
    status:        Mapped[str]      = mapped_column(String(20), default="running")   # running | completed | stopped | error
    algorithm:     Mapped[str]      = mapped_column(String(20), default="DQN")
    config:        Mapped[dict]     = mapped_column(JSON,        nullable=False)
    total_episodes:Mapped[int]      = mapped_column(Integer,     nullable=False)
    best_reward:   Mapped[float]    = mapped_column(Float,       default=0.0)
    created_at:    Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at:   Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    episodes: Mapped[list[EpisodeLog]] = relationship(back_populates="session", cascade="all, delete-orphan")


class EpisodeLog(Base):
    """One row per training episode — drives the dashboard charts."""

    __tablename__ = "episode_logs"

    id:          Mapped[int]   = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id:  Mapped[str]   = mapped_column(String(36), ForeignKey("training_sessions.id", ondelete="CASCADE"))
    episode:     Mapped[int]   = mapped_column(Integer, nullable=False)
    reward:      Mapped[float] = mapped_column(Float,   nullable=False)
    avg_wait:    Mapped[float] = mapped_column(Float,   nullable=False)
    throughput:  Mapped[float] = mapped_column(Float,   nullable=False)
    epsilon:     Mapped[float] = mapped_column(Float,   nullable=False)
    loss:        Mapped[float | None] = mapped_column(Float, nullable=True)
    duration_s:  Mapped[float] = mapped_column(Float,   nullable=False)

    session: Mapped[TrainingSession] = relationship(back_populates="episodes")
