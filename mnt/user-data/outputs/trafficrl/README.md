# TrafficRL — Deep Q-Network Traffic Signal Optimiser

A production-ready FastAPI backend that trains a DQN agent to minimise vehicle
waiting time at a signalised intersection in real time, streaming episode
metrics to the dashboard over WebSocket.

---

## Architecture

```
trafficrl/
├── app/
│   ├── main.py               # FastAPI app, CORS, lifecycle
│   ├── config.py             # Pydantic settings (.env)
│   ├── api/
│   │   ├── schemas.py        # Request / response models
│   │   ├── training.py       # POST /train, POST /train/stop, GET /status, WS /ws/training
│   │   └── predict.py        # POST /predict
│   ├── rl/
│   │   ├── environment.py    # TrafficSignalEnv (gym-style)
│   │   ├── network.py        # DQNNetwork (MLP, LayerNorm)
│   │   ├── replay_buffer.py  # ReplayBuffer (deque, uniform sampling)
│   │   └── agent.py          # DQNAgent (epsilon-greedy, target net, Huber loss)
│   └── db/
│       ├── database.py       # SQLAlchemy async engine + session
│       └── models.py         # TrainingSession, EpisodeLog
├── alembic/                  # DB migrations
├── tests/
│   └── test_dqn.py           # 14 unit tests (env, buffer, network, agent)
├── docker-compose.yml        # API + PostgreSQL
└── Dockerfile
```

---

## DQN Implementation

### State space
```
[north_queue, south_queue, east_queue, west_queue]
```
Each value is normalised to `[0, 1]` by dividing by `max_queue` (default 20).

### Action space
```
0 → NS green (North + South lanes clear, East + West wait)
1 → EW green (East + West lanes clear, North + South wait)
```

### Reward
```
r(t) = −Σ(queue lengths)           # negative total waiting vehicles
     − yellow_penalty               # if phase changed this step (default 2.0)
```

### Key components

| Component | Detail |
|---|---|
| **Network** | 3-layer MLP: `4 → 128 → 128 → 64 → 2`, LayerNorm + ReLU, Kaiming init |
| **Replay Buffer** | Circular deque, capacity 10 000, uniform random sampling |
| **Epsilon-greedy** | Linear decay: `ε = max(ε_end, ε × decay)` each episode |
| **Target network** | Hard sync every `target_update_freq` episodes (default 10) |
| **Loss** | Huber (SmoothL1) — robust to outlier Q-value errors |
| **Optimiser** | Adam, gradient clipped at norm 10 |
| **Min green** | Phase switches blocked for `min_green_steps` (default 3) |

---

## Quick start

### With Docker (recommended)

```bash
cp .env.example .env
docker compose up --build
```

API is live at `http://localhost:8000`. Docs at `http://localhost:8000/docs`.

### Without Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Start Postgres (or set DATABASE_URL to your instance)
cp .env.example .env

# Run migrations
alembic upgrade head

# Start API
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## API Reference

### `POST /train`
Start a DQN training session.

**Request body** (all fields optional — defaults shown):
```json
{
  "algorithm":          "DQN",
  "episodes":           500,
  "max_steps":          200,
  "learning_rate":      0.0003,
  "gamma":              0.99,
  "epsilon_start":      1.0,
  "epsilon_end":        0.05,
  "epsilon_decay":      0.995,
  "batch_size":         64,
  "buffer_capacity":    10000,
  "target_update_freq": 10,
  "hidden_size":        128,
  "arrival_rate_ns":    0.3,
  "arrival_rate_ew":    0.3,
  "throughput_ns":      3,
  "throughput_ew":      3,
  "min_green_steps":    3
}
```

**Response** `202`:
```json
{
  "session_id": "uuid",
  "status":     "running",
  "message":    "DQN training started for 500 episodes."
}
```

---

### `POST /train/stop`
Gracefully stop the current training session.

```json
{ "status": "stopped", "session_id": "uuid" }
```

---

### `GET /status`
Poll current training state.

```json
{
  "status":         "running",
  "session_id":     "uuid",
  "episode":        142,
  "total_episodes": 500,
  "reward":         -183.4,
  "avg_wait":       6.12,
  "throughput":     2.8,
  "epsilon":        0.494,
  "best_reward":    -161.2,
  "loss":           0.00412
}
```

`status` is one of: `idle | running | completed | stopped | error`

---

### `POST /predict`
Get the optimal action for a given intersection state.

```json
{ "state": [5, 3, 8, 2], "normalize": true }
```

`state` = `[north, south, east, west]` queue lengths (raw vehicle counts).  
`normalize: true` → divides by 20 before inference (matches training).

**Response**:
```json
{
  "action":      1,
  "action_name": "EW green",
  "q_values":    [-12.4, -8.7],
  "epsilon":     0.05
}
```

---

### `WS /ws/training`
Real-time episode stream. Connect before or after `POST /train`.

**On connect**:
```json
{ "type": "connected", "status": "running", "session_id": "uuid" }
```

**Each episode**:
```json
{
  "type":           "episode",
  "session_id":     "uuid",
  "episode":        50,
  "total_episodes": 500,
  "reward":         -201.3,
  "avg_wait":       8.42,
  "throughput":     2.6,
  "epsilon":        0.778,
  "best_reward":    -187.1,
  "loss":           0.00538,
  "duration_s":     0.021
}
```

**On completion**:
```json
{ "type": "done", "session_id": "uuid", "status": "completed" }
```

---

## Running tests

```bash
pip install pytest pytest-asyncio
pytest tests/test_dqn.py -v
```

14 tests covering:
- Environment: reset, step, both actions, invalid action, termination, NS/EW throughput, min green enforcement, queue saturation, reward bounds
- ReplayBuffer: capacity, push, sample shapes, dtype, undersized sample error
- DQNNetwork: output shape, batch shape, finite outputs, gradient flow
- DQNAgent: action selection, epsilon-greedy exploration/exploitation, learn before/after buffer fill, target sync, predict, full training loop, stop signal
