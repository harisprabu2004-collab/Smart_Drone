
import os
import math
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


BATTERY_CAPACITY_MAH    = 5000      
SAFE_RETURN_THRESHOLD   = 25.0     
CRITICAL_BATTERY_PCT    = 20.0  
DRONE_CRUISE_SPEED_MS   = 8.0      
MODEL_PATH              = os.path.join(os.path.dirname(__file__), "ml", "model.pkl")
DATA_PATH               = os.path.join(os.path.dirname(__file__), "data", "flight_log.csv")



def generate_training_data(n_samples: int = 2000) -> pd.DataFrame:
    
    np.random.seed(42)
    rows = []

    for _ in range(n_samples):
        battery_start   = np.random.uniform(30, 100)
        flight_time     = np.random.uniform(0, 1200)         
        speed           = np.random.uniform(2, 15)
        altitude        = np.random.uniform(5, 100)
        current_draw    = np.random.uniform(8, 25)         
        distance        = speed * flight_time

        drain_pct   = (current_draw * (flight_time / 3600) / (BATTERY_CAPACITY_MAH / 1000)) * 100
        battery_now = max(0, battery_start - drain_pct)

        voltage = 10.5 + (battery_now / 100) * 2.1

        drain_per_metre = current_draw / (speed * (BATTERY_CAPACITY_MAH / 1000) * 3600 / 100)

        usable_battery  = max(0, battery_now - SAFE_RETURN_THRESHOLD)
        max_range_m     = usable_battery / drain_per_metre if drain_per_metre > 0 else 0

        return_drain    = (current_draw * (distance / speed / 3600) / (BATTERY_CAPACITY_MAH / 1000)) * 100
        battery_return  = max(0, battery_now - return_drain)

        rows.append({
            "battery_pct"       : round(battery_now, 2),
            "voltage"           : round(voltage, 3),
            "current_amps"      : round(current_draw, 2),
            "speed_ms"          : round(speed, 2),
            "altitude_m"        : round(altitude, 2),
            "flight_time_s"     : round(flight_time, 1),
            "distance_m"        : round(distance, 1),
            "max_range_m"       : round(max_range_m, 1),
            "battery_on_return" : round(battery_return, 2),
            "can_return_safe"   : int(battery_return >= SAFE_RETURN_THRESHOLD)
        })

    return pd.DataFrame(rows)


FEATURE_COLS = [
    "battery_pct", "voltage", "current_amps",
    "speed_ms", "altitude_m", "flight_time_s", "distance_m"
]


class BatteryMLModel:

    def __init__(self):
        self.range_model        = self._make_pipeline()
        self.return_batt_model  = self._make_pipeline()
        self.safe_model         = self._make_pipeline()
        self._trained           = False
        self._metrics           = {}
        self._history           = []          # live telemetry log
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(DATA_PATH),  exist_ok=True)

    @staticmethod
    def _make_pipeline() -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("lr",     LinearRegression())
        ])

    def train(self, df: pd.DataFrame = None):
        """Train all three models. Uses synthetic data if df is None."""
        if df is None:
            df = generate_training_data(2000)

        X = df[FEATURE_COLS].values
        y_range     = df["max_range_m"].values
        y_ret_batt  = df["battery_on_return"].values
        y_safe      = df["can_return_safe"].values

        X_tr, X_te, yr_tr, yr_te, yb_tr, yb_te, ys_tr, ys_te = train_test_split(
            X, y_range, y_ret_batt, y_safe, test_size=0.2, random_state=42
        )

        self.range_model.fit(X_tr, yr_tr)
        self.return_batt_model.fit(X_tr, yb_tr)
        self.safe_model.fit(X_tr, ys_tr)

        # Evaluate
        self._metrics = {
            "range_mae"     : round(mean_absolute_error(yr_te, self.range_model.predict(X_te)), 2),
            "range_r2"      : round(r2_score(yr_te, self.range_model.predict(X_te)), 4),
            "ret_batt_mae"  : round(mean_absolute_error(yb_te, self.return_batt_model.predict(X_te)), 2),
            "ret_batt_r2"   : round(r2_score(yb_te, self.return_batt_model.predict(X_te)), 4),
            "safe_acc"      : round(float(np.mean(
                                (self.safe_model.predict(X_te) >= 0.5).astype(int) == ys_te.astype(int)
                            )), 4),
            "train_samples" : len(df)
        }

        self._trained = True
        self._save()
        print(f"[ML] ✅ Model trained. Range MAE={self._metrics['range_mae']}m  "
              f"BattReturn MAE={self._metrics['ret_batt_mae']}%  "
              f"SafeAcc={self._metrics['safe_acc']}")
        return self._metrics

    def _save(self):
        joblib.dump({
            "range_model"       : self.range_model,
            "return_batt_model" : self.return_batt_model,
            "safe_model"        : self.safe_model,
            "metrics"           : self._metrics
        }, MODEL_PATH)
        print(f"[ML] Model saved → {MODEL_PATH}")

    def load(self):
        if os.path.exists(MODEL_PATH):
            bundle = joblib.load(MODEL_PATH)
            self.range_model        = bundle["range_model"]
            self.return_batt_model  = bundle["return_batt_model"]
            self.safe_model         = bundle["safe_model"]
            self._metrics           = bundle.get("metrics", {})
            self._trained           = True
            print(f"[ML] Model loaded from {MODEL_PATH}")
            return True
        return False

    def log_telemetry(self, telemetry: dict):
        """Append a live telemetry snapshot to the history buffer."""
        self._history.append({
            "battery_pct"   : telemetry.get("battery_percentage", 0),
            "voltage"       : telemetry.get("voltage", 0),
            "current_amps"  : telemetry.get("current", 0),
            "speed_ms"      : telemetry.get("speed_ms", 0),
            "altitude_m"    : telemetry.get("altitude_m", 0),
            "flight_time_s" : telemetry.get("flight_time_s", 0),
            "distance_m"    : telemetry.get("distance_m", 0),
        })
        if len(self._history) >= 50 and len(self._history) % 50 == 0:
            self._retrain_with_live_data()

    def _retrain_with_live_data(self):
        """Blend synthetic + real flight data and retrain."""
        synthetic = generate_training_data(500)
        live_df   = pd.DataFrame(self._history)

        def estimate_range(row):
            dp = row["current_amps"] / max(row["speed_ms"], 0.1) / (BATTERY_CAPACITY_MAH / 1000) / 3600 * 100
            usable = max(0, row["battery_pct"] - SAFE_RETURN_THRESHOLD)
            return usable / dp if dp > 0 else 0

        live_df["max_range_m"]       = live_df.apply(estimate_range, axis=1)
        live_df["battery_on_return"] = live_df["battery_pct"] - (live_df["current_amps"] * 0.1)
        live_df["can_return_safe"]   = (live_df["battery_on_return"] >= SAFE_RETURN_THRESHOLD).astype(int)

        combined = pd.concat([synthetic, live_df], ignore_index=True)
        self.train(combined)
        print(f"[ML] Retrained with {len(combined)} samples ({len(live_df)} real)")

    def predict(self, telemetry: dict) -> dict:
        """
        Main prediction entry point.
        Returns a rich dictionary with all predictions and warnings.
        """
        if not self._trained:
            return self._fallback_predict(telemetry)

        batt    = telemetry.get("battery_percentage", 50)
        volt    = telemetry.get("voltage", 11.5)
        amps    = telemetry.get("current", 10)
        speed   = telemetry.get("speed_ms", 5)
        alt     = telemetry.get("altitude_m", 0)
        ftime   = telemetry.get("flight_time_s", 0)
        dist    = telemetry.get("distance_m", 0)

        X = np.array([[batt, volt, amps, speed, alt, ftime, dist]])

        max_range_m     = max(0, float(self.range_model.predict(X)[0]))
        batt_on_return  = float(self.return_batt_model.predict(X)[0])
        safe_score      = float(self.safe_model.predict(X)[0])
        can_return      = safe_score >= 0.5

        batt_on_return  = round(min(100, max(0, batt_on_return)), 1)
        max_range_m     = round(max_range_m, 1)
        max_range_km    = round(max_range_m / 1000, 3)

        warnings = []
        if batt <= CRITICAL_BATTERY_PCT:
            warnings.append({
                "level"   : "CRITICAL",
                "code"    : "BATT_CRITICAL",
                "message" : f"⚠️ CRITICAL! Battery at {batt:.1f}%. Land immediately!"
            })
        elif batt <= SAFE_RETURN_THRESHOLD:
            warnings.append({
                "level"   : "WARNING",
                "code"    : "BATT_LOW",
                "message" : f"⚠️ Battery below {SAFE_RETURN_THRESHOLD}%. Return to home now."
            })
        if not can_return:
            warnings.append({
                "level"   : "WARNING",
                "code"    : "UNSAFE_RETURN",
                "message" : "⚠️ Drone may NOT safely return with current battery."
            })

        drain_rate_pct_per_min = (amps / (BATTERY_CAPACITY_MAH / 1000)) * (100 / 60) if amps > 0 else 1
        minutes_remaining = max(0, (batt - SAFE_RETURN_THRESHOLD) / drain_rate_pct_per_min)

        return {
            "battery_percentage"    : round(batt, 1),
            "max_range_m"           : max_range_m,
            "max_range_km"          : max_range_km,
            "battery_on_return_pct" : batt_on_return,
            "can_return_safe"       : can_return,
            "safe_score"            : round(safe_score, 3),
            "minutes_remaining"     : round(minutes_remaining, 1),
            "warnings"              : warnings,
            "model_metrics"         : self._metrics
        }

    def preflight_check(self, telemetry: dict, planned_distance_m: float) -> dict:
        """
        Check if the drone can complete a planned round trip.
        planned_distance_m: one-way distance the user wants to fly.
        """
        prediction      = self.predict(telemetry)
        round_trip_m    = planned_distance_m * 2
        max_range_m     = prediction["max_range_m"]
        batt            = telemetry.get("battery_percentage", 0)

        can_complete    = round_trip_m <= max_range_m
        margin_m        = max_range_m - round_trip_m

        safety_margin   = (margin_m / max(round_trip_m, 1)) * 100

        status = "SAFE"
        if not can_complete:
            status = "UNSAFE"
        elif safety_margin < 20:
            status = "CAUTION"

        result = {
            "status"                : status,
            "can_complete_mission"  : can_complete,
            "planned_one_way_m"     : planned_distance_m,
            "planned_round_trip_m"  : round_trip_m,
            "max_range_m"           : max_range_m,
            "margin_m"              : round(margin_m, 1),
            "safety_margin_pct"     : round(safety_margin, 1),
            "current_battery_pct"   : round(batt, 1),
            "prediction"            : prediction,
            "pre_flight_warnings"   : []
        }

        if status == "UNSAFE":
            result["pre_flight_warnings"].append({
                "level"   : "DANGER",
                "message" : (
                    f"🚫 DO NOT TAKE OFF! Round trip requires {round_trip_m:.0f}m "
                    f"but max range is only {max_range_m:.0f}m. "
                    f"Charge battery to at least {min(100, batt + abs(margin_m/100)):.0f}%."
                )
            })
        elif status == "CAUTION":
            result["pre_flight_warnings"].append({
                "level"   : "CAUTION",
                "message" : (
                    f"⚠️ CAUTION! Only {safety_margin:.0f}% safety margin. "
                    f"Consider charging more before flying {planned_distance_m:.0f}m."
                )
            })
        else:
            result["pre_flight_warnings"].append({
                "level"   : "OK",
                "message" : (
                    f"✅ SAFE TO FLY. Battery sufficient for {planned_distance_m:.0f}m round trip "
                    f"with {safety_margin:.0f}% margin ({margin_m:.0f}m extra range)."
                )
            })

        if batt <= SAFE_RETURN_THRESHOLD:
            result["pre_flight_warnings"].insert(0, {
                "level"   : "DANGER",
                "message" : f"🔋 Battery at {batt:.1f}%! Minimum {SAFE_RETURN_THRESHOLD}% required. CHARGE NOW."
            })

        return result

    def _fallback_predict(self, telemetry: dict) -> dict:
        """Physics-based prediction when ML model isn't trained yet."""
        batt    = telemetry.get("battery_percentage", 50)
        amps    = max(telemetry.get("current", 10), 0.1)
        speed   = max(telemetry.get("speed_ms", 5), 0.1)

        drain_per_metre = amps / (speed * (BATTERY_CAPACITY_MAH / 1000) * 3600 / 100)
        usable          = max(0, batt - SAFE_RETURN_THRESHOLD)
        max_range_m     = usable / drain_per_metre if drain_per_metre > 0 else 0

        return {
            "battery_percentage"    : round(batt, 1),
            "max_range_m"           : round(max_range_m, 1),
            "max_range_km"          : round(max_range_m / 1000, 3),
            "battery_on_return_pct" : round(max(0, batt - usable), 1),
            "can_return_safe"       : batt > SAFE_RETURN_THRESHOLD,
            "safe_score"            : batt / 100,
            "minutes_remaining"     : 0,
            "warnings"              : [],
            "model_metrics"         : {}
        }


_model_instance = None

def get_model() -> BatteryMLModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = BatteryMLModel()
        if not _model_instance.load():
            print("[ML] No saved model found. Training from scratch...")
            _model_instance.train()
    return _model_instance
