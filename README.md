# TrafficRL — DQN Traffic Signal Optimiser

A complete end-to-end system: a Deep Q-Network that learns to minimise vehicle waiting time at a signalised intersection, with a live dashboard that visualises training in real time.

```
┌─────────────────────────────────┐      ┌───────────────────────────┐
│         dashboard.html          │      │        server.py           │
│   (runs in any browser)         │      │   (Flask, port 5000)       │
│                                 │      │                            │
│  [Train] → POST /train ─────────┼──────┼→ spawns background thread │
│  poll GET /status every 1 s ───┼──────┼→ reads TrainingState       │
│  charts animate in real time    │      │                            │
│  signal sim ← POST /predict ───┼──────┼→ QNetwork.forward()       │
└─────────────────────────────────┘      └───────────────────────────┘
```

---

## Quick start — Simulation Mode (no backend needed)

Open `dashboard.html` in a browser. It runs in **Simulation Mode** by default — a full DQN training curve is generated locally in JavaScript at 500 episodes / 50 seconds. Every chart, signal, and log row works exactly as it would with a real backend.

The header shows **⚡ Sim Mode** and the status pill shows **Connected**.

---

## Full stack — Flask backend + real DQN

### 1 · Install

```bash
python3 -m venv .venv && source .venv/bin/activate

# CPU-only torch (faster install, ~200 MB)
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
pip install flask==3.0.3 flask-cors==4.0.1 gunicorn==22.0.0 numpy==1.26.4
```

Or use the provided script:

```bash
chmod +x start.sh && ./start.sh
```

### 2 · Run the backend

```bash
python server.py              # http://localhost:5000
python server.py --port 8080  # custom port
```

### 3 · Connect the frontend

In `dashboard.html`, find the config block near the top of the `<script>` tag and make two edits:

```js
const API_BASE        = 'http://localhost:5000';  // your backend URL
const SIMULATION_MODE = false;                    // flip this
```

Refresh the browser. The header badge disappears and the status pill shows **Connected**.

---

## Architecture

### Backend — `server.py`

| Component | Detail |
|---|---|
| Framework | Flask 3 + flask-cors |
| Transport | HTTP polling — GET /status every 1 s |
| Training | Background `threading.Thread` (daemon) |
| State | `TrainingState` dataclass protected by `threading.Lock` |
| Q-network | Inline `QNetwork` (4 → 128 → LayerNorm → 128 → 64 → 2) |
| Replay buffer | `app/rl/replay_buffer.py` (circular deque, uniform sampling) |
| Environment | `app/rl/environment.py` (Poisson arrivals, 4-arm queues) |

#### Endpoints

```
GET  /health
     → { service, status: "ok" }

GET  /status
     → { connected, status, episode, total_episodes,
          reward, avg_wait, throughput, epsilon, loss,
          best_reward, session_id }

POST /train
Body (all optional):
     { algorithm, episodes=500, max_steps=200, learning_rate=0.0003,
       gamma=0.99, epsilon_start=1.0, epsilon_end=0.05,
       epsilon_decay=0.995, batch_size=64, buffer_capacity=10000,
       target_update_freq=10, hidden_size=128,
       arrival_rate_ns=0.3, arrival_rate_ew=0.3,
       throughput_ns=3, throughput_ew=3, min_green_steps=3 }
     → 202 { session_id, status, message, episodes, max_steps }

POST /train/stop
     → { status: "stopping", session_id }

POST /predict
Body: { state: [north, south, east, west], normalize: true }
     → { action: 0|1, action_name, q_values, epsilon }
```

`action 0` = NS green. `action 1` = EW green.

### Frontend — `dashboard.html`

Single HTML file — no build step, no npm, no framework.

| Layer | Detail |
|---|---|
| Charts | Chart.js 4 — Reward, Avg Wait, Epsilon with MA-10 overlay |
| Simulation | Canvas intersection with 4 directional queues |
| Polling | `apiFetch` wrapper → `setInterval` 1000 ms |
| Signal control | `dqnPredictTick` every 500 ms → POST /predict |
| Error handling | `LogThrottle` class — suppresses repeated errors with cooldown |
| Sim mode | `simulatedApiFetch` mirrors all 4 endpoints locally in JS |

#### Switching modes — one line change

```js
// At the top of the <script> block:
const API_BASE        = 'https://your-app.onrender.com';
const SIMULATION_MODE = false;
```

Every fetch call, chart, signal, and log row uses the same code path in both modes.

### DQN implementation

```
State:    [north, south, east, west]  — normalised queue lengths (0–1)
Actions:  0 = NS green  |  1 = EW green
Reward:   −(total waiting vehicles) − yellow_penalty on phase change

Network:      4 → 128 (LayerNorm+ReLU) → 128 (LayerNorm+ReLU) → 64 → 2
Loss:         Huber (SmoothL1)
Optimiser:    Adam, lr=3e-4, gradient clip at norm 10
Exploration:  ε-greedy, ε × 0.995 per episode, floor 0.05
Target net:   hard sync every 10 episodes
Replay buf:   10 000 transitions, uniform random sampling
Min hold:     2 000 ms before phase can switch (prevents visual flicker)
```

---

## Deployment on Render

### 1 · Push to GitHub

```bash
git init
git add dashboard.html server.py requirements_flask.txt Procfile render.yaml start.sh app/
git commit -m "TrafficRL initial commit"
git remote add origin https://github.com/YOU/trafficrl.git
git push -u origin main
```

### 2 · Create a Web Service on Render

**Option A — automatic (recommended):** Render detects `render.yaml` and configures everything automatically. Just connect your repo and click Deploy.

**Option B — manual:**

1. New → Web Service → connect your GitHub repo
2. **Runtime:** Python 3
3. **Build command:**
   ```
   pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu && pip install flask==3.0.3 flask-cors==4.0.1 gunicorn==22.0.0 numpy==1.26.4
   ```
4. **Start command:**
   ```
   gunicorn server:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120
   ```
5. **Health check path:** `/health`

### 3 · Connect the frontend

Once deployed, copy the service URL from Render's dashboard (e.g. `https://trafficrl-api.onrender.com`) and update the frontend:

```js
const API_BASE        = 'https://trafficrl-api.onrender.com';
const SIMULATION_MODE = false;
```

### Free tier limitations

- Services **spin down after 15 minutes of inactivity** — the first request after sleep takes ~30 s. The dashboard's page-load health check wakes the server automatically.
- Training state is **in-memory** — a Render restart clears it. The frontend detects `status: idle` on next poll and shows "Connected".
- For persistent history, use the FastAPI backend (`app/main.py`) with a PostgreSQL add-on.

---

## Running tests

```bash
pip install pytest pytest-asyncio httpx aiosqlite
pytest tests/test_dqn.py -v    # 14 unit tests: env, buffer, network, agent
pytest tests/test_api.py  -v   # 22 integration tests: all HTTP endpoints
```

---

## Project layout

```
trafficrl/
├── dashboard.html          Frontend — open in any browser, zero dependencies
├── server.py               Flask backend — single file, self-contained
├── requirements_flask.txt  Flask dependencies (includes gunicorn)
├── Procfile                gunicorn start command for Render / Heroku
├── render.yaml             Render deployment manifest
├── start.sh                Local dev startup script
│
├── app/
│   ├── rl/
│   │   ├── environment.py      TrafficSignalEnv (Gymnasium-compatible)
│   │   ├── replay_buffer.py    Circular buffer, uniform sampling
│   │   ├── network.py          DQNNetwork MLP
│   │   └── agent.py            Full async DQN agent (used by FastAPI)
│   ├── api/                    FastAPI routers (training, predict, sessions)
│   ├── db/                     SQLAlchemy models + Alembic migrations
│   └── main.py                 FastAPI app (production alternative)
│
├── tests/
│   ├── test_dqn.py             Unit tests: env, buffer, network, agent
│   └── test_api.py             Integration tests: all endpoints
│
├── Dockerfile                  Container build for the FastAPI backend
└── docker-compose.yml          FastAPI + PostgreSQL for local development
```

---

## Two backends, one frontend

| | `server.py` (Flask) | `app/main.py` (FastAPI) |
|---|---|---|
| Framework | Flask 3 | FastAPI + Uvicorn |
| Database | None — in-memory | PostgreSQL via SQLAlchemy |
| Real-time | HTTP polling | WebSocket + polling |
| Training | `threading.Thread` | `asyncio` task |
| History | Lost on restart | Persisted in DB |
| Deploy | Render free tier | Docker / Render paid |
| Best for | Demo, prototype | Production |

To use the FastAPI backend:
```bash
pip install -r requirements.txt                 # installs FastAPI, uvicorn, etc.
docker compose up                               # starts FastAPI + PostgreSQL
# or without Docker:
uvicorn app.main:app --port 5000 --reload
```
Then in `dashboard.html`: `API_BASE = 'http://localhost:5000'`, `SIMULATION_MODE = false`.
