from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ── POST /train ──────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    algorithm:          str   = Field("DQN",    description="Algorithm name (DQN supported)")
    episodes:           int   = Field(500,      ge=1,    le=10_000)
    max_steps:          int   = Field(200,      ge=10,   le=2_000)
    learning_rate:      float = Field(3e-4,     gt=0,    lt=1.0)
    gamma:              float = Field(0.99,     ge=0.0,  le=1.0)
    epsilon_start:      float = Field(1.0,      ge=0.0,  le=1.0)
    epsilon_end:        float = Field(0.05,     ge=0.0,  le=1.0)
    epsilon_decay:      float = Field(0.995,    gt=0.0,  lt=1.0)
    batch_size:         int   = Field(64,       ge=8,    le=512)
    buffer_capacity:    int   = Field(10_000,   ge=100)
    target_update_freq: int   = Field(10,       ge=1)
    hidden_size:        int   = Field(128,      ge=16)

    # Environment
    arrival_rate_ns:    float = Field(0.3,  ge=0.0)
    arrival_rate_ew:    float = Field(0.3,  ge=0.0)
    throughput_ns:      int   = Field(3,    ge=1)
    throughput_ew:      int   = Field(3,    ge=1)
    min_green_steps:    int   = Field(3,    ge=1)


class TrainResponse(BaseModel):
    session_id: str
    status:     str
    message:    str


# ── POST /predict ────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    state: list[float] = Field(
        ...,
        min_length=4, max_length=4,
        description="[north, south, east, west] queue lengths (raw counts)",
        examples=[[5, 3, 8, 2]],
    )
    normalize: bool = Field(True, description="Divide by max_queue (20) before inference")


class PredictResponse(BaseModel):
    action:      int
    action_name: str
    q_values:    list[float]
    epsilon:     float


# ── GET /status ──────────────────────────────────────────────────────────────

class StatusResponse(BaseModel):
    status:         str                   # idle | running | completed | stopped | error
    session_id:     Optional[str]  = None
    episode:        Optional[int]  = None
    total_episodes: Optional[int]  = None
    reward:         Optional[float]= None
    avg_wait:       Optional[float]= None
    throughput:     Optional[float]= None
    epsilon:        Optional[float]= None
    best_reward:    Optional[float]= None
    loss:           Optional[float]= None


# ── WebSocket episode message ─────────────────────────────────────────────────

class EpisodeMessage(BaseModel):
    type:           str = "episode"
    session_id:     str
    episode:        int
    total_episodes: int
    reward:         float
    avg_wait:       float
    throughput:     float
    epsilon:        float
    best_reward:    float
    loss:           Optional[float] = None
    duration_s:     float
