from pathlib import Path

from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

from image_pipeline import analyse_image
from model_registry import build_registry_from_env


app = Flask(__name__)
registry = build_registry_from_env()
MODELS_DIR = Path(app.config.get("MODELS_DIR", "/app/models"))
ALLOWED_TASK_TYPES = {"tabular", "image"}


@app.get("/health")
def health():
    """Return service health and active model info."""
    return jsonify(
        {
            "status": "ok",
            "active_model_name": registry.active_model_name,
            "registered_models": len(registry.list_models().get("models", {})),
        }
    )


@app.get("/models")
def list_models():
    """Return all registered models and the active model name."""
    return jsonify(registry.list_models())


@app.post("/models/upload")
def upload_model():
    """Accept a model file upload and register it in the registry."""
    model_file = request.files.get("model")
    task_type = request.form.get("task_type", "tabular").strip().lower()
    display_name = request.form.get("display_name")

    if not model_file or not model_file.filename:
        return jsonify(
            {"error": "Missing model file under form field 'model'"}
        ), 400
    if task_type not in ALLOWED_TASK_TYPES:
        return jsonify(
            {"error": "Unsupported task_type. Use one of: "
             f"{sorted(ALLOWED_TASK_TYPES)}"}
        ), 400

    filename = secure_filename(model_file.filename)
    destination = Path("/app/models") / filename
    model_file.save(destination)

    try:
        metadata = registry.register_model(
            filename=filename, task_type=task_type, display_name=display_name)
    except ValueError as exc:
        destination.unlink(missing_ok=True)
        return jsonify({"error": str(exc)}), 400

    return jsonify({"message": "Model uploaded", "metadata": metadata}), 201


@app.post("/models/select")
def select_model():
    """Switch the active model by name."""
    payload = request.get_json(silent=True) or {}
    model_name = payload.get("model_name")
    if not model_name:
        return jsonify({"error": "Missing 'model_name' in JSON body"}), 400

    print(f"DEBUG: Starting select_model for {model_name}", flush=True)
    try:
        result = registry.select_model(model_name)
        print(f"DEBUG: registry.select_model finished", flush=True)
    except Exception as exc:
        print(f"DEBUG: Exception in select_model: {exc}", flush=True)
        return jsonify({"error": str(exc)}), 400

    return jsonify({"message": "Active model updated", **result})


@app.post("/predict/tabular")
def predict_tabular():
    """Run tabular prediction against the active model."""
    payload = request.get_json(silent=True) or {}
    features = payload.get("features")
    if not isinstance(features, list) or not features:
        return jsonify(
            {"error": "Provide 'features' as a non-empty JSON list"}
        ), 400

    try:
        numeric_features = [float(x) for x in features]
        result = registry.predict_tabular(numeric_features)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(result)


@app.post("/predict/image")
def predict_image():
    """Run image prediction using the active image model or heuristics."""
    image_file = request.files.get("image")
    if not image_file or not image_file.filename:
        return jsonify(
            {"error": "Missing image file under form field 'image'"}
        ), 400

    image_bytes = image_file.read()
    if not image_bytes:
        return jsonify({"error": "Uploaded image is empty"}), 400

    try:
        model = None
        if registry.active_model_name:
            try:
                model = registry.load_model(registry.active_model_name)
            except Exception as e:
                print(
                    f"DEBUG: Could not load active model "
                    f"{registry.active_model_name}: {e}",
                    flush=True
                )

        result = analyse_image(image_bytes, model=model)
    except Exception as exc:
        return jsonify({"error": f"Image analysis failed: {exc}"}), 400

    return jsonify(result)


@app.get("/xai/active-model")
def explain_active_model():
    """Return XAI explanation for the active model."""
    try:
        explanation = registry.explain_active_model()
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(explanation)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
