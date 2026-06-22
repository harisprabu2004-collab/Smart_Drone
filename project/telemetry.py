"""
telemetry.py
============
Connects to ArduPilot SITL simulator via MAVLink protocol.
Reads live: battery voltage, current, battery %, GPS, speed, altitude, armed state.

HOW TO RUN ArduPilot SITL BEFORE STARTING THIS:
  sim_vehicle.py -v ArduCopter --console --map
  (This opens MAVLink on UDP port 14550 by default)
"""

import threading
import time
import math
import random
from datetime import datetime

# ── Try to import pymavlink; fall back to simulator if not installed ──────────
try:
    from pymavlink import mavutil
    MAVLINK_AVAILABLE = True
except ImportError:
    MAVLINK_AVAILABLE = False
    print("[WARN] pymavlink not installed. Running in SIMULATION mode.")


# ─────────────────────────────────────────────────────────────────────────────
#  SIMULATED TELEMETRY  (used when ArduPilot is not running)
# ─────────────────────────────────────────────────────────────────────────────
class SimulatedDrone:
    """Generates realistic fake telemetry so the UI works without ArduPilot."""

    def __init__(self):
        self._battery_pct   = 100.0
        self._voltage       = 12.6          # Fully charged 3S LiPo
        self._current       = 0.0
        self._speed         = 0.0
        self._altitude      = 0.0
        self._distance      = 0.0
        self._armed         = False
        self._lat           = 11.0168       # Tiruppur, Tamil Nadu
        self._lon           = 77.0498
        self._flight_time   = 0             # seconds
        self._tick          = 0
        self._thread        = threading.Thread(target=self._simulate, daemon=True)
        self._thread.start()

    def _simulate(self):
        while True:
            self._tick += 1
            t = self._tick

            # Simulate arming after 5 seconds
            if t == 5:
                self._armed = True

            if self._armed:
                # Simulate takeoff, hover, fly, return
                phase = (t % 120)           # 120-second flight cycle

                if phase < 15:              # Takeoff
                    self._altitude  = min(30, self._altitude + 2)
                    self._speed     = 3.0
                    self._current   = 18.0

                elif phase < 80:            # Flying
                    self._altitude  = 30 + random.uniform(-1, 1)
                    self._speed     = 8.0 + random.uniform(-1, 1)
                    self._current   = 15.0 + random.uniform(-2, 2)
                    self._distance += self._speed * 0.5   # metres per tick

                elif phase < 100:           # Return / Land
                    self._altitude  = max(0, self._altitude - 1.5)
                    self._speed     = 5.0
                    self._current   = 12.0

                else:                       # Landed
                    self._altitude  = 0
                    self._speed     = 0
                    self._current   = 1.5   # idle draw

                # Drain battery
                drain_rate  = (self._current * 0.05)   # % per second at this current
                self._battery_pct = max(0, self._battery_pct - drain_rate * 0.1)
                self._voltage     = 10.5 + (self._battery_pct / 100) * 2.1
                self._flight_time += 1

                # Move GPS position slightly
                self._lat += random.uniform(-0.00001, 0.00001)
                self._lon += random.uniform(-0.00001, 0.00001)

            time.sleep(1)

    def get_telemetry(self):
        return {
            "battery_percentage" : round(self._battery_pct, 1),
            "voltage"            : round(self._voltage, 2),
            "current"            : round(self._current, 1),
            "speed_ms"           : round(self._speed, 2),
            "altitude_m"         : round(self._altitude, 1),
            "distance_m"         : round(self._distance, 1),
            "armed"              : self._armed,
            "latitude"           : round(self._lat, 6),
            "longitude"          : round(self._lon, 6),
            "flight_time_s"      : self._flight_time,
            "timestamp"          : datetime.now().strftime("%H:%M:%S"),
            "source"             : "SIMULATED"
        }


# ─────────────────────────────────────────────────────────────────────────────
#  REAL MAVLINK TELEMETRY  (used when ArduPilot SITL is running)
# ─────────────────────────────────────────────────────────────────────────────
class MAVLinkTelemetry:
    """
    Connects to ArduPilot via UDP and continuously reads telemetry.
    Default ArduPilot SITL address: udp:127.0.0.1:14550
    """

    def __init__(self, connection_string="udp:127.0.0.1:14550"):
        self.connection_string = connection_string
        self._data   = {}
        self._lock   = threading.Lock()
        self._conn   = None
        self._running = False
        self._connect()

    def _connect(self):
        print(f"[MAVLink] Connecting to {self.connection_string} ...")
        try:
            self._conn    = mavutil.mavlink_connection(self.connection_string)
            self._conn.wait_heartbeat(timeout=10)
            print(f"[MAVLink] ✅ Connected! System {self._conn.target_system}, "
                  f"Component {self._conn.target_component}")
            self._running = True
            t = threading.Thread(target=self._read_loop, daemon=True)
            t.start()
        except Exception as e:
            print(f"[MAVLink] ❌ Connection failed: {e}")
            print("[MAVLink] Falling back to simulation mode.")
            self._running = False

    def _read_loop(self):
        """Continuously read MAVLink messages and update internal state."""
        state = {
            "battery_percentage" : 0.0,
            "voltage"            : 0.0,
            "current"            : 0.0,
            "speed_ms"           : 0.0,
            "altitude_m"         : 0.0,
            "distance_m"         : 0.0,
            "armed"              : False,
            "latitude"           : 0.0,
            "longitude"          : 0.0,
            "flight_time_s"      : 0,
            "timestamp"          : "",
            "source"             : "MAVLINK"
        }
        start_time = time.time()

        while self._running:
            try:
                msg = self._conn.recv_match(blocking=True, timeout=1)
                if msg is None:
                    continue

                msg_type = msg.get_type()

                # ── Battery Status ─────────────────────────────────────────
                if msg_type == "SYS_STATUS":
                    state["voltage"]            = msg.voltage_battery / 1000.0
                    state["current"]            = msg.current_battery / 100.0
                    state["battery_percentage"] = msg.battery_remaining

                # ── GPS / Position ─────────────────────────────────────────
                elif msg_type == "GLOBAL_POSITION_INT":
                    state["latitude"]   = msg.lat / 1e7
                    state["longitude"]  = msg.lon / 1e7
                    state["altitude_m"] = msg.relative_alt / 1000.0

                # ── Speed ──────────────────────────────────────────────────
                elif msg_type == "VFR_HUD":
                    state["speed_ms"]   = msg.groundspeed
                    state["altitude_m"] = msg.alt

                # ── Armed State ────────────────────────────────────────────
                elif msg_type == "HEARTBEAT":
                    armed = (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0
                    state["armed"] = armed

                state["flight_time_s"] = int(time.time() - start_time)
                state["timestamp"]     = datetime.now().strftime("%H:%M:%S")

                with self._lock:
                    self._data = dict(state)

            except Exception as e:
                print(f"[MAVLink] Read error: {e}")
                time.sleep(0.5)

    def get_telemetry(self):
        with self._lock:
            return dict(self._data) if self._data else {}

    @property
    def is_connected(self):
        return self._running


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC FACTORY — auto-selects real or simulated
# ─────────────────────────────────────────────────────────────────────────────
def create_telemetry_source(connection_string="udp:127.0.0.1:14550"):
    """
    Returns a telemetry object with a .get_telemetry() method.
    Tries real MAVLink first; falls back to simulation automatically.
    """
    if MAVLINK_AVAILABLE:
        src = MAVLinkTelemetry(connection_string)
        if src.is_connected:
            return src
    print("[Telemetry] Using SimulatedDrone for development/demo.")
    return SimulatedDrone()
