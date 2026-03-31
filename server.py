"""
Flask backend — DQN Traffic Signal Optimiser
============================================
Runs on http://localhost:5000

Endpoints
---------
GET  /status    — current training state (polled by frontend)
POST /train     — start training in background thread
POST /train/stop — stop the running training session
POST /predict   — greedy action for a given intersection state

The training loop runs in a daemon thread so it never blocks the
Flask request handlers. All shared state lives in the `TrainingState`
dataclass and is protected by a threading.Lock.
"""

from __future__ import annotations

import copy
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, jsonify, request
from flask_cors import CORS

# ── Re-use existing RL modules from the app package ──────────────────────────
# Adjust sys.path so `app.*` imports resolve when running from project root.
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from app.rl.environment  import TrafficSignalEnv, TrafficEnvConfig
from app.rl.replay_buffer import ReplayBuffer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("trafficrl.flask")

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})   # allow any origin in dev

# ---------------------------------------------------------------------------
# Q-Network  (lightweight 3-layer MLP, no external dependency on app.rl.network)
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """
    Maps state [N, S, E, W] → Q-values for [NS-green, EW-green].

    Architecture: 4 → 128 → 128 → 64 → 2
    LayerNorm on hidden layers for training stability.
    """

    def __init__(self, state_dim: int = 4, action_dim: int = 2, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden,    hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden,    64),                           nn.ReLU(),
            nn.Linear(64,        action_dim),
        )
        # Kaiming init for all linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ---------------------------------------------------------------------------
# Shared training state  (thread-safe via a lock)
# ---------------------------------------------------------------------------

@dataclass
class TrainingState:
    # Status flags
    training:       bool  = False
    status:         str   = "idle"      # idle | running | completed | stopped | error
    session_id:     Optional[str] = None
    error_msg:      Optional[str] = None

    # Per-episode metrics (updated by background thread, read by /status)
    episode:        int   = 0
    total_episodes: int   = 0
    reward:         float = 0.0
    avg_wait:       float = 0.0
    throughput:     float = 0.0
    epsilon:        float = 1.0
    loss:           Optional[float] = None
    best_reward:    float = float("-inf")
    duration_s:     float = 0.0

    # Stop signal (set by POST /train/stop)
    _stop:          bool  = field(default=False, repr=False)

    # Lock for cross-thread access
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self) -> dict:
        """Return a JSON-serialisable snapshot for /status."""
        with self._lock:
            return {
                "connected":      True,
                "status":         self.status,
                "training":       self.training,
                "session_id":     self.session_id,
                "episode":        self.episode,
                "total_episodes": self.total_episodes,
                "reward":         round(self.reward,      3),
                "avg_wait":       round(self.avg_wait,    3),
                "throughput":     round(self.throughput,  3),
                "epsilon":        round(self.epsilon,     4),
                "loss":           round(self.loss, 5) if self.loss is not None else None,
                "best_reward":    round(self.best_reward, 3),
                "duration_s":     round(self.duration_s,  3),
                "error":          self.error_msg,
            }

    def update(self, **kwargs) -> None:
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def request_stop(self) -> None:
        with self._lock:
            self._stop = True

    def should_stop(self) -> bool:
        with self._lock:
            return self._stop


# Single global state object
state = TrainingState()

# ---------------------------------------------------------------------------
# DQN training — runs in a background daemon thread
# ---------------------------------------------------------------------------

def run_training(
    episodes:           int   = 500,
    max_steps:          int   = 200,
    learning_rate:      float = 3e-4,
    gamma:              float = 0.99,
    epsilon_start:      float = 1.0,
    epsilon_end:        float = 0.05,
    epsilon_decay:      float = 0.995,
    batch_size:         int   = 64,
    buffer_capacity:    int   = 10_000,
    min_buffer_size:    int   = 500,
    target_update_freq: int   = 10,
    hidden_size:        int   = 128,
    # Environment config
    arrival_rate_ns:    float = 0.3,
    arrival_rate_ew:    float = 0.3,
    throughput_ns:      int   = 3,
    throughput_ew:      int   = 3,
    min_green_steps:    int   = 3,
) -> None:
    """
    Full DQN training loop.
    Updates the global `state` object after every episode so /status
    always reflects current progress without any database.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training thread started on device=%s  episodes=%d", device, episodes)

    # ── Build networks ────────────────────────────────────────────────────────
    online_net = QNetwork(state_dim=4, action_dim=2, hidden=hidden_size).to(device)
    target_net = copy.deepcopy(online_net).to(device)
    target_net.eval()

    optimiser = optim.Adam(online_net.parameters(), lr=learning_rate)
    loss_fn   = nn.SmoothL1Loss()   # Huber loss
    buffer    = ReplayBuffer(capacity=buffer_capacity)

    epsilon     = epsilon_start
    best_reward = float("-inf")

    # ── Environment ──────────────────────────────────────────────────────────
    env_cfg = TrafficEnvConfig(
        max_steps       = max_steps,
        arrival_rate_ns = arrival_rate_ns,
        arrival_rate_ew = arrival_rate_ew,
        throughput_ns   = throughput_ns,
        throughput_ew   = throughput_ew,
        min_green_steps = min_green_steps,
    )
    env = TrafficSignalEnv(env_cfg)

    state.update(total_episodes=episodes, status="running", training=True)

    # ── Episode loop ─────────────────────────────────────────────────────────
    for ep in range(1, episodes + 1):

        # Check for stop request before each episode
        if state.should_stop():
            logger.info("Stop requested — halting after episode %d", ep - 1)
            state.update(status="stopped", training=False)
            return

        t0           = time.perf_counter()
        obs, _       = env.reset(seed=ep)
        ep_reward    = 0.0
        ep_wait_sum  = 0.0
        ep_thru_sum  = 0.0
        ep_loss_sum  = 0.0
        ep_loss_cnt  = 0
        last_info    = {}

        # ── Step loop ─────────────────────────────────────────────────────
        for step in range(max_steps):

            # ε-greedy action selection
            if np.random.random() < epsilon:
                action = np.random.randint(2)
            else:
                with torch.no_grad():
                    t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    action = int(online_net(t).argmax(dim=1).item())

            next_obs, reward, terminated, truncated, info = env.step(action)
            buffer.push(obs, action, reward, next_obs, terminated or truncated)

            obs          = next_obs
            ep_reward   += reward
            ep_wait_sum += info.get("avg_wait",  0.0)
            ep_thru_sum += info.get("throughput", 0.0)
            last_info    = info

            # ── Gradient update ───────────────────────────────────────────
            if len(buffer) >= min_buffer_size:
                try:
                    s_b, a_b, r_b, ns_b, d_b = buffer.sample(batch_size)
                    s_b  = s_b.to(device);   a_b  = a_b.to(device)
                    r_b  = r_b.to(device);   ns_b = ns_b.to(device)
                    d_b  = d_b.to(device)

                    current_q = online_net(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        max_next_q = target_net(ns_b).max(dim=1).values
                        target_q   = r_b + gamma * max_next_q * (1.0 - d_b)

                    loss = loss_fn(current_q, target_q)
                    optimiser.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=10.0)
                    optimiser.step()

                    ep_loss_sum += loss.item()
                    ep_loss_cnt += 1
                except ValueError:
                    pass   # buffer not full enough yet

            if terminated or truncated:
                break

        # ── Post-episode bookkeeping ─────────────────────────────────────
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if ep % target_update_freq == 0:
            target_net.load_state_dict(online_net.state_dict())

        if ep_reward > best_reward:
            best_reward = ep_reward

        steps_done = last_info.get("step", max_steps)
        avg_wait   = ep_wait_sum / max(steps_done, 1)
        avg_thru   = ep_thru_sum / max(steps_done, 1)
        avg_loss   = ep_loss_sum / ep_loss_cnt if ep_loss_cnt > 0 else None
        duration   = time.perf_counter() - t0

        # ── Update shared state (frontend reads this via /status) ────────
        state.update(
            episode     = ep,
            reward      = round(ep_reward,  3),
            avg_wait    = round(avg_wait,   3),
            throughput  = round(avg_thru,   3),
            epsilon     = round(epsilon,    4),
            loss        = round(avg_loss, 5) if avg_loss is not None else None,
            best_reward = round(best_reward, 3),
            duration_s  = round(duration,   3),
        )

        logger.info(
            "Ep %d/%d | R=%.1f | wait=%.2f | ε=%.3f | loss=%s",
            ep, episodes, ep_reward, avg_wait, epsilon,
            f"{avg_loss:.5f}" if avg_loss else "—",
        )

        # Yield the GIL briefly so Flask threads can respond
        time.sleep(0)

    # ── Training finished ─────────────────────────────────────────────────
    logger.info("Training complete after %d episodes", episodes)
    state.update(status="completed", training=False)

# ---------------------------------------------------------------------------
# Endpoint helpers
# ---------------------------------------------------------------------------

def _bad(msg: str, code: int = 400):
    return jsonify({"error": msg}), code

# ---------------------------------------------------------------------------
# GET /status
# ---------------------------------------------------------------------------

@app.route("/status", methods=["GET"])
def get_status():
    """
    Return current training state.

    Shape:
        {
          "connected": true,
          "status": "idle" | "running" | "completed" | "stopped" | "error",
          "training": bool,
          "episode": int,
          "total_episodes": int,
          "reward": float,
          "avg_wait": float,
          "throughput": float,
          "epsilon": float,
          "loss": float | null,
          "best_reward": float,
          "session_id": str | null
        }
    """
    return jsonify(state.snapshot()), 200

# ---------------------------------------------------------------------------
# POST /train
# ---------------------------------------------------------------------------

@app.route("/train", methods=["POST"])
def post_train():
    """
    Start a new DQN training session in a background thread.
    Returns 409 if a session is already running.

    Request body (all fields optional — defaults shown):
        {
          "algorithm": "DQN",
          "episodes": 500,
          "max_steps": 200,
          "learning_rate": 0.0003,
          "gamma": 0.99,
          "epsilon_start": 1.0,
          "epsilon_end": 0.05,
          "epsilon_decay": 0.995,
          "batch_size": 64,
          "buffer_capacity": 10000,
          "min_buffer_size": 500,
          "target_update_freq": 10,
          "hidden_size": 128,
          "arrival_rate_ns": 0.3,
          "arrival_rate_ew": 0.3,
          "throughput_ns": 3,
          "throughput_ew": 3,
          "min_green_steps": 3
        }
    """
    if state.training:
        return _bad("Training already running. POST /train/stop first.", 409)

    body = request.get_json(silent=True) or {}

    # Extract and validate params with safe defaults
    def _int(key, default, lo=1, hi=100_000):
        v = body.get(key, default)
        try:    v = int(v)
        except: return default
        return max(lo, min(hi, v))

    def _float(key, default, lo=0.0, hi=1.0):
        v = body.get(key, default)
        try:    v = float(v)
        except: return default
        return max(lo, min(hi, v))

    params = dict(
        episodes           = _int  ("episodes",           500,   1,   10_000),
        max_steps          = _int  ("max_steps",          200,   10,  2_000),
        learning_rate      = _float("learning_rate",      3e-4,  1e-6, 1.0),
        gamma              = _float("gamma",              0.99,  0.5,  1.0),
        epsilon_start      = _float("epsilon_start",      1.0,   0.0,  1.0),
        epsilon_end        = _float("epsilon_end",        0.05,  0.0,  1.0),
        epsilon_decay      = _float("epsilon_decay",      0.995, 0.8,  1.0),
        batch_size         = _int  ("batch_size",         64,    4,    512),
        buffer_capacity    = _int  ("buffer_capacity",    10_000, 100, 1_000_000),
        min_buffer_size    = _int  ("min_buffer_size",    500,   1,    10_000),
        target_update_freq = _int  ("target_update_freq", 10,    1,    1_000),
        hidden_size        = _int  ("hidden_size",        128,   16,   1_024),
        arrival_rate_ns    = _float("arrival_rate_ns",    0.3,   0.0,  10.0),
        arrival_rate_ew    = _float("arrival_rate_ew",    0.3,   0.0,  10.0),
        throughput_ns      = _int  ("throughput_ns",      3,     1,    20),
        throughput_ew      = _int  ("throughput_ew",      3,     1,    20),
        min_green_steps    = _int  ("min_green_steps",    3,     1,    100),
    )

    session_id = str(uuid.uuid4())
    state.update(
        training       = True,
        status         = "running",
        session_id     = session_id,
        error_msg      = None,
        episode        = 0,
        total_episodes = params["episodes"],
        reward         = 0.0,
        avg_wait       = 0.0,
        throughput     = 0.0,
        epsilon        = params["epsilon_start"],
        loss           = None,
        best_reward    = float("-inf"),
        _stop          = False,
    )

    t = threading.Thread(target=_training_wrapper, kwargs=params, daemon=True)
    t.start()

    logger.info("Training session %s started — %d episodes", session_id, params["episodes"])
    return jsonify({
        "session_id": session_id,
        "status":     "running",
        "message":    f"DQN training started for {params['episodes']} episodes.",
        **params,
    }), 202


def _training_wrapper(**kwargs):
    """Wraps run_training to catch and record unexpected exceptions."""
    try:
        run_training(**kwargs)
    except Exception as exc:
        logger.exception("Training thread crashed: %s", exc)
        state.update(status="error", training=False, error_msg=str(exc))

# ---------------------------------------------------------------------------
# POST /train/stop
# ---------------------------------------------------------------------------

@app.route("/train/stop", methods=["POST"])
def post_train_stop():
    """Signal the background training thread to stop after the current episode."""
    if not state.training:
        return jsonify({"status": "idle", "message": "No training session active."}), 200

    state.request_stop()
    logger.info("Stop requested for session %s", state.session_id)
    return jsonify({
        "status":     "stopping",
        "session_id": state.session_id,
        "message":    "Stop signal sent — will halt after current episode.",
    }), 200

# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------

# Module-level Q-network for inference — lazily initialised on first call,
# then reused.  Protected by a lock so two simultaneous requests don't race
# on initialisation.
_predict_net: Optional[QNetwork]     = None
_predict_lock: threading.Lock        = threading.Lock()
_predict_device: Optional[torch.device] = None

def _get_predict_net() -> tuple[QNetwork, torch.device]:
    global _predict_net, _predict_device
    with _predict_lock:
        if _predict_net is None:
            _predict_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _predict_net    = QNetwork().to(_predict_device)
            logger.info("Inference network initialised on %s", _predict_device)
    return _predict_net, _predict_device


@app.route("/predict", methods=["POST"])
def post_predict():
    """
    Return the greedy action for a given intersection state.

    Request:
        { "state": [north, south, east, west], "normalize": true }

    Response:
        { "action": 0 | 1, "action_name": "NS green" | "EW green",
          "q_values": [float, float], "epsilon": float }
    """
    body = request.get_json(silent=True)
    if not body or "state" not in body:
        return _bad('Missing "state" field. Expected: {"state": [N, S, E, W]}')

    raw_state = body["state"]
    if not isinstance(raw_state, list) or len(raw_state) != 4:
        return _bad('"state" must be a list of exactly 4 numbers: [N, S, E, W]')

    try:
        arr = np.array(raw_state, dtype=np.float32)
    except (TypeError, ValueError):
        return _bad('"state" values must all be numbers')

    # Normalise to [0, 1] to match training (default: divide by max_queue=20)
    if body.get("normalize", True):
        arr = np.clip(arr / 20.0, 0.0, 1.0)

    net, device = _get_predict_net()
    with torch.no_grad():
        t      = torch.FloatTensor(arr).unsqueeze(0).to(device)
        q_vals = net(t).squeeze(0)
        action = int(q_vals.argmax().item())

    action_names = {0: "NS green", 1: "EW green"}
    return jsonify({
        "action":      action,
        "action_name": action_names[action],
        "q_values":    q_vals.cpu().tolist(),
        "epsilon":     round(state.epsilon, 4),
    }), 200

# ---------------------------------------------------------------------------
# CORS pre-flight (Flask-CORS handles this, but keep an explicit handler
# so browsers never see a 405 on OPTIONS)
# ---------------------------------------------------------------------------

@app.route("/train",      methods=["OPTIONS"])
@app.route("/train/stop", methods=["OPTIONS"])
@app.route("/predict",    methods=["OPTIONS"])
@app.route("/status",     methods=["OPTIONS"])
def options_handler():
    return "", 204

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"service": "TrafficRL-Flask", "status": "ok"}), 200

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TrafficRL Flask backend")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5000)),
                        help="Port to listen on (default: 5000, or $PORT env var)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    args = parser.parse_args()

    logger.info("Starting TrafficRL Flask server on http://%s:%d", args.host, args.port)
    logger.info("Endpoints: GET /status  POST /train  POST /train/stop  POST /predict")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
