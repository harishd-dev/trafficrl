"""
Inference endpoint — POST /predict
===================================
Returns the greedy action (and Q-values) for a given intersection state.
Can be called during or after training.
"""

from __future__ import annotations

import logging

import numpy as np
from fastapi import APIRouter, HTTPException

from app.api.schemas import PredictRequest, PredictResponse

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_QUEUE = 20   # must match TrafficEnvConfig.max_queue


@router.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    # Import here to avoid circular deps with training.py
    from app.api.training import _agent

    if _agent is None:
        raise HTTPException(
            404,
            "No trained agent found. Start training first via POST /train.",
        )

    state = np.array(req.state, dtype=np.float32)

    if req.normalize:
        state = np.clip(state / MAX_QUEUE, 0.0, 1.0)

    result = _agent.predict(state)

    return PredictResponse(
        action      = result["action"],
        action_name = result["action_name"],
        q_values    = result["q_values"],
        epsilon     = result["epsilon"],
    )
