import os

import requests
from flask import Flask, jsonify, request, render_template, Response


app = Flask(__name__)
AI_SERVICE_URL = os.getenv("AI_SERVICE_URL", "http://ai-service:8000")
TIMEOUT = 30


@app.get("/health")
def health():
    """Web-app self health (used by Docker healthcheck)."""
    return jsonify({"status": "ok", "ai_service_url": AI_SERVICE_URL})


@app.get("/")
def index():
    """Render single-page UI."""
    return render_template("index.html", ai_service_url=AI_SERVICE_URL)



def _relay(resp: requests.Response):
    """Return upstream response as-is (JSON or text) without crashing on non-JSON bodies."""
    ctype = resp.headers.get("Content-Type", "")
    if "application/json" in ctype:
        try:
            return jsonify(resp.json()), resp.status_code
        except Exception:
            # Malformed JSON; fall back to raw text
            pass
    return Response(resp.content, status=resp.status_code, mimetype=ctype or "text/plain")


@app.get("/ai/health")
def ai_health():
    """Proxy to AI service health endpoint."""
    response = requests.get(f"{AI_SERVICE_URL}/health", timeout=TIMEOUT)
    return _relay(response)


@app.get("/models")
def models():
    response = requests.get(f"{AI_SERVICE_URL}/models", timeout=TIMEOUT)
    return _relay(response)


@app.post("/models/upload")
def upload_model():
    model_file = request.files.get("model")
    if not model_file:
        return jsonify({"error": "Missing model file under form field 'model'"}), 400

    files = {
        "model": (model_file.filename, model_file.stream, model_file.mimetype or "application/octet-stream"),
    }
    data = {
        "task_type": request.form.get("task_type", "tabular"),
        "display_name": request.form.get("display_name", model_file.filename),
    }
    response = requests.post(f"{AI_SERVICE_URL}/models/upload", files=files, data=data, timeout=TIMEOUT)
    return _relay(response)


@app.post("/models/select")
def select_model():
    payload = request.get_json(silent=True) or {}
    response = requests.post(f"{AI_SERVICE_URL}/models/select", json=payload, timeout=TIMEOUT)
    return _relay(response)


@app.post("/predict/tabular")
def predict_tabular():
    payload = request.get_json(silent=True) or {}
    response = requests.post(f"{AI_SERVICE_URL}/predict/tabular", json=payload, timeout=TIMEOUT)
    return _relay(response)


@app.post("/predict/image")
def predict_image():
    image_file = request.files.get("image")
    if not image_file:
        return jsonify({"error": "Missing image file under form field 'image'"}), 400

    files = {
        "image": (image_file.filename, image_file.stream, image_file.mimetype or "application/octet-stream"),
    }
    response = requests.post(f"{AI_SERVICE_URL}/predict/image", files=files, timeout=TIMEOUT)
    return _relay(response)


@app.get("/xai/active-model")
def active_model_xai():
    response = requests.get(f"{AI_SERVICE_URL}/xai/active-model", timeout=TIMEOUT)
    return _relay(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
