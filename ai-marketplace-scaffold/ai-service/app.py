from pathlib import Path

from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

from image_pipeline import analyse_image
from model_registry import build_registry_from_env
from utils.generate_masks import warmup_rembg


app = Flask(__name__)
registry = build_registry_from_env()
MODELS_DIR = Path(app.config.get("MODELS_DIR", "/app/models"))
ALLOWED_TASK_TYPES = {"tabular", "image"}


def _startup_warmup():
    """Pre-initialise heavy resources so the first user request isn't cold."""
    try:
        warmup_rembg()
    except Exception as exc:
        print(f"[startup] rembg warmup failed (non-critical): {exc}", flush=True)
    try:
        from image_pipeline import _load_colour_refs
        _load_colour_refs()
    except Exception as exc:
        print(f"[startup] colour refs warmup failed: {exc}", flush=True)
    if registry.active_model_name:
        try:
            registry.load_model(registry.active_model_name)
            print(f"[startup] pre-loaded model '{registry.active_model_name}'", flush=True)
        except Exception as exc:
            print(f"[startup] model pre-load failed: {exc}", flush=True)


_startup_warmup()


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


@app.get("/models/architectures")
def list_architectures():
    """Return all valid torchvision model names."""
    from torchvision.models import list_models
    return jsonify({"architectures": sorted(list_models())})


@app.post("/models/validate-arch")
def validate_arch():
    """Check whether an architecture name is a valid torchvision model."""
    payload = request.get_json(silent=True) or {}
    arch = payload.get("architecture", "").strip()
    if not arch:
        return jsonify({"valid": False, "error": "Architecture name required"}), 400
    try:
        from torchvision.models import get_model
        get_model(arch, weights=None)
        return jsonify({"valid": True})
    except Exception:
        return jsonify({"valid": False, "error": f"Unknown architecture: '{arch}'"}), 400


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

    image_config = None
    if task_type == "image":
        architecture = request.form.get("architecture", "").strip()
        if not architecture:
            return jsonify({"error": "Field 'architecture' is required for image models"}), 400
        is_mtl = request.form.get("is_mtl", "false").lower() == "true"
        image_config = {"architecture": architecture, "is_mtl": is_mtl}

    filename = secure_filename(model_file.filename)
    destination = Path("/app/models") / filename
    model_file.save(destination)

    if image_config:
        try:
            # Auto-detect produce type count before full load
            if image_config["is_mtl"]:
                suffix = destination.suffix.lower()
                if suffix == ".safetensors":
                    from safetensors import safe_open
                    with safe_open(str(destination), framework="pt", device="cpu") as f:
                        num_produce_types = f.get_slice("type_head.1.weight").get_shape()[0]
                else:
                    import torch
                    sd = torch.load(str(destination), map_location="cpu", weights_only=True)
                    num_produce_types = sd["type_head.1.weight"].shape[0]
                image_config["num_produce_types"] = int(num_produce_types)
                print(f"[upload] Auto-detected num_produce_types={num_produce_types}", flush=True)
        except KeyError:
            destination.unlink(missing_ok=True)
            return jsonify({"error": "Not a valid MTL model — 'type_head' not found in weights. "
                            "Uncheck Multi-task if this is an STL model."}), 400
        except Exception as exc:
            destination.unlink(missing_ok=True)
            return jsonify({"error": f"Could not read weights: {exc}"}), 400

        try:
            # Full load validates architecture + weight shape compatibility
            from image_pipeline import load_model
            loaded = load_model(use_file=True, path=str(destination), config=image_config)
            if loaded is None:
                raise ValueError("Model could not be loaded")
            print(f"[upload] Model load validation passed for '{filename}'", flush=True)
        except RuntimeError:
            destination.unlink(missing_ok=True)
            arch = image_config["architecture"]
            return jsonify({"error": f"Weight mismatch: the saved weights are incompatible with '{arch}'. "
                            f"Make sure the architecture matches what the model was trained with."}), 400
        except Exception as exc:
            destination.unlink(missing_ok=True)
            return jsonify({"error": f"Model validation failed: {exc}"}), 400

    try:
        metadata = registry.register_model(
            filename=filename, task_type=task_type, display_name=display_name,
            image_config=image_config)
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
        if registry.active_model_name:
            state = registry._read_registry()
            meta = state.get("models", {}).get(registry.active_model_name, {})
            result["model_display_name"] = meta.get("display_name", registry.active_model_name)
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
