"""
TrafficRL — FastAPI application entrypoint
==========================================
Run with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.training import router as training_router
from app.api.predict  import router as predict_router
from app.api.sessions import router as sessions_router
from app.db           import init_db
from app.config       import get_settings

settings = get_settings()

logging.basicConfig(
    level   = settings.log_level,
    format  = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt = "%H:%M:%S",
)

app = FastAPI(
    title       = "TrafficRL API",
    description = "Deep Q-Network traffic signal timing optimiser",
    version     = "1.0.0",
)

# ── CORS — allow the dashboard (any origin in dev) ──────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # tighten in production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Routes ───────────────────────────────────────────────────────────────────
app.include_router(training_router, tags=["training"])
app.include_router(predict_router,  tags=["inference"])
app.include_router(sessions_router)


# ── Lifecycle ─────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup() -> None:
    await init_db()
    logging.getLogger(__name__).info("Database tables ensured.")


@app.get("/", tags=["health"])
async def root() -> dict:
    return {"service": "TrafficRL", "status": "ok", "docs": "/docs"}


@app.get("/health", tags=["health"])
async def health() -> dict:
    return {"status": "ok"}
