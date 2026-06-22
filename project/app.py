
import threading
import time
import json
from collections import deque
from flask import Flask, jsonify, request, render_template, Response
from flask_cors import CORS

from telemetry import create_telemetry_source
from ml_model  import get_model

app  = Flask(__name__)
CORS(app)


print("[App] Initialising telemetry source...")
telemetry_source = create_telemetry_source("udp:127.0.0.1:14550")


print("[App] Loading ML model...")
model = get_model()


history = deque(maxlen=300)

def _background_logger():
    while True:
        try:
            telem = telemetry_source.get_telemetry()
            if telem:
                model.log_telemetry(telem)
                history.append(telem)
        except Exception as e:
            print(f"[Logger] Error: {e}")
        time.sleep(1)

_logger_thread = threading.Thread(target=_background_logger, daemon=True)
_logger_thread.start()
print("[App] Background logger started.")



@app.route("/")
def index():
    """Serve the dashboard HTML page."""
    return render_template("index.html")


@app.route("/api/telemetry")
def api_telemetry():
    """Return the latest raw telemetry snapshot."""
    telem = telemetry_source.get_telemetry()
    return jsonify({"success": True, "data": telem})


@app.route("/api/predict")
def api_predict():
    """Return ML predictions based on current telemetry."""
    telem      = telemetry_source.get_telemetry()
    prediction = model.predict(telem)
    return jsonify({"success": True, "data": prediction})


@app.route("/api/preflight", methods=["POST"])
def api_preflight():
    """
    Pre-flight safety check.
    Body: { "distance_m": <float>  }   (one-way distance in metres)
    """
    body = request.get_json(force=True, silent=True) or {}
    distance_m = float(body.get("distance_m", 0))

    if distance_m <= 0:
        return jsonify({"success": False, "error": "distance_m must be > 0"}), 400

    telem  = telemetry_source.get_telemetry()
    result = model.preflight_check(telem, distance_m)
    return jsonify({"success": True, "data": result})


@app.route("/api/history")
def api_history():
    """Return last N telemetry snapshots (default 60)."""
    n = min(int(request.args.get("n", 60)), 300)
    return jsonify({"success": True, "data": list(history)[-n:]})


@app.route("/api/model/metrics")
def api_model_metrics():
    """Return ML model training metrics."""
    return jsonify({"success": True, "data": model._metrics})


@app.route("/api/model/retrain", methods=["POST"])
def api_retrain():
    """Manually trigger model retraining."""
    metrics = model.train()
    return jsonify({"success": True, "data": metrics})


@app.route("/api/stream")
def api_stream():
    """
    Server-Sent Events stream — pushes telemetry + predictions every second.
    JavaScript can subscribe with:  const es = new EventSource('/api/stream')
    """
    def generate():
        while True:
            telem = telemetry_source.get_telemetry()
            pred  = model.predict(telem)
            payload = json.dumps({"telemetry": telem, "prediction": pred})
            yield f"data: {payload}\n\n"
            time.sleep(1)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  🚁  Smart Drone Battery Management System")
    print("  🌐  Dashboard → http://localhost:5000")
    print("="*60 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
