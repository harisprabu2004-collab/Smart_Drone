# 🚁 Smart Drone Battery Management System
## Complete Setup & Run Guide

---

## PROJECT STRUCTURE

```
drone_bms/
├── app.py              ← Flask server (run this)
├── telemetry.py        ← ArduPilot MAVLink connection
├── ml_model.py         ← Linear Regression ML model
├── requirements.txt    ← Python packages
├── templates/
│   └── index.html      ← Dashboard UI
├── ml/
│   └── model.pkl       ← Saved model (auto-created)
└── data/
    └── flight_log.csv  ← Flight history (auto-created)
```

---

## STEP 1 — Install Python & Packages

Make sure you have Python 3.9 or higher installed.

Open terminal / command prompt in the `drone_bms` folder:

```bash
pip install -r requirements.txt
```

---

## STEP 2 — Install ArduPilot SITL (Simulator)

### Option A — Linux / WSL (Recommended)
```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y git python3 python3-pip

# Clone ArduPilot
git clone https://github.com/ArduPilot/ardupilot.git
cd ardupilot
git submodule update --init --recursive

# Install ArduPilot dependencies
Tools/environment_install/install-prereqs-ubuntu.sh -y

# Build ArduCopter SITL
./waf configure --board sitl
./waf copter
```

### Option B — Use Mission Planner (Windows)
1. Download Mission Planner from https://ardupilot.org/planner/docs/mission-planner-installation.html
2. Open Mission Planner → Simulation tab → click "COPTER"
3. This starts SITL automatically on UDP port 14550

---

## STEP 3 — Run ArduPilot SITL

```bash
# In the ardupilot directory:
cd ArduCopter
sim_vehicle.py -v ArduCopter --console --map

# This opens MAVLink on:   UDP 127.0.0.1:14550  ← our app connects here
```

You will see a map and a console window — the simulator is now running.

---

## STEP 4 — Run the Dashboard

```bash
# In the drone_bms folder:
python app.py
```

Open browser: **http://localhost:5000**

---

## STEP 5 — Using the Dashboard

### Battery Gauge (Left)
- Shows live battery percentage as a circular gauge
- Green = OK, Yellow = below 50%, Orange = below 25%, Red = CRITICAL
- Shows voltage, current draw, time remaining, max range

### ⚠️ Battery Warning
- When battery drops below **25%** → orange warning appears
- When battery drops below **20%** → full-screen RED CRITICAL ALERT

### Predictions (Center)
- **Max Range** — how far the drone can travel with current battery
- **Battery on Return** — predicted battery % when it gets back
- **Time Remaining** — estimated flight minutes left
- **Safety Score** — ML confidence that return is safe
- Live telemetry: speed, altitude, GPS, flight time

### Pre-Flight Check (Bottom)
1. Enter the distance you want to fly (one-way, in metres or km)
2. Click **RUN CHECK**
3. System predicts if drone can complete the round trip:
   - ✅ SAFE — enough battery with margin
   - ⚠️ CAUTION — possible but tight
   - 🚫 UNSAFE — DO NOT TAKE OFF shown before takeoff

### Warnings Panel (Right)
- Live warnings from ML model
- Model accuracy metrics
- Retrain button to update model with new data

---

## HOW THE ML MODEL WORKS

### Algorithm: Linear Regression (scikit-learn)
Three separate models are trained:

| Model | Input Features | Output |
|-------|---------------|--------|
| Range Model | battery%, voltage, current, speed, altitude, time, distance | Max range (m) |
| Return Battery Model | same | Battery % on return |
| Safety Model | same | Can return safely (0/1) |

### Training Data
- **Initial training**: 2,000 synthetic samples generated from drone physics equations
- **Live retraining**: Every 50 real telemetry readings, model retrains with combined real + synthetic data
- Model saved to `ml/model.pkl` automatically

### Physics Formula Used
```
drain_per_metre = current_amps / (speed × battery_capacity_Ah × 3600 / 100)
max_range = usable_battery_pct / drain_per_metre
usable_battery = battery_now - safe_return_threshold (25%)
```

---

## CONFIGURATION (edit ml_model.py)

```python
BATTERY_CAPACITY_MAH    = 5000   # Change to your battery (mAh)
SAFE_RETURN_THRESHOLD   = 25.0   # % minimum to safely land
CRITICAL_BATTERY_PCT    = 20.0   # % triggers full-screen alert
DRONE_CRUISE_SPEED_MS   = 8.0    # m/s average speed
```

---

## RUNNING WITHOUT ARDUPILOT (Demo Mode)

If ArduPilot is not installed, the system automatically switches to
**Simulation Mode** — it generates realistic fake telemetry so you can
develop and test the UI without hardware.

You will see `SIMULATED` in the status bar at the top right.

---

## API ENDPOINTS

| Method | URL | Description |
|--------|-----|-------------|
| GET | `/` | Dashboard UI |
| GET | `/api/telemetry` | Latest raw telemetry |
| GET | `/api/predict` | ML predictions |
| POST | `/api/preflight` | Pre-flight check `{"distance_m": 500}` |
| GET | `/api/history?n=60` | Last 60 telemetry snapshots |
| GET | `/api/model/metrics` | ML model accuracy |
| POST | `/api/model/retrain` | Retrain the model |
| GET | `/api/stream` | Server-Sent Events live stream |

---

## TROUBLESHOOTING

**"pymavlink not installed"**
→ Run: `pip install pymavlink`

**"Connection failed: [Errno 111]"**
→ ArduPilot SITL is not running. Start it first (Step 3), OR the app will auto-switch to Simulation mode.

**Dashboard not loading**
→ Make sure Flask is running: `python app.py`
→ Check http://localhost:5000 (not 8000 or 3000)

**Model accuracy is low**
→ Click "RETRAIN MODEL" in the warnings panel after flying for a few minutes.

---

## FINAL YEAR PROJECT REPORT — KEY POINTS

1. **MAVLink Protocol** — Industry standard for drone communication
2. **Real-time telemetry** — 1 Hz polling, <1s latency
3. **Online learning** — Model retrains every 50 samples with real data
4. **Physics + ML hybrid** — Linear regression augments physics equations
5. **Pre-flight safety** — Prevents takeoff when battery is insufficient
6. **Critical alert system** — 25% warning + 20% full-screen critical alert
