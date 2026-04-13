import json
import os
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

import joblib

from bootstrap_model import bootstrap_default_model


SUPPORTED_EXECUTABLE_TABULAR_SUFFIXES = {".joblib", ".pkl"}
SUPPORTED_ACCEPTED_SUFFIXES = {".joblib", ".pkl", ".safetensors", ".gguf"}


class DummyTabularModel:
    """Fallback model used when use_file_models=False.

    Produces deterministic but simple outputs for demo/testing without requiring a real file-backed model.
    """

    def predict(self, X):
        # Binary-ish split by sum threshold; returns 0/1 based on mean
        out = []
        for row in X:
            s = sum(float(x) for x in row)
            out.append(1 if s >= 2.5 else 0)
        return out

    def predict_proba(self, X):
        # Pseudo probabilities for two classes based on sigmoid of sum
        import math

        probs = []
        for row in X:
            s = sum(float(x) for x in row)
            p1 = 1 / (1 + math.exp(-s))
            p0 = 1 - p1
            probs.append([p0, p1])
        return probs


def load_model(use_file: bool = True, path: Optional[str] = None):
    """Load a tabular model.

    - If use_file is True and path is provided, load via joblib.
    - If use_file is False, return a DummyTabularModel instance.
    """
    if not use_file:
        return DummyTabularModel()
    if not path:
        raise ValueError("Model path must be provided when use_file=True")
    return joblib.load(Path(path))


class ModelRegistry:
    def __init__(self, models_dir: str, registry_path: str, default_model_name: str, use_file_models: bool = True):
        self.models_dir = Path(models_dir)
        self.registry_path = Path(registry_path)
        self.default_model_name = default_model_name
        self.use_file_models = use_file_models
        self._lock = Lock()
        self._loaded_models: Dict[str, Any] = {}
        self.active_model_name: Optional[str] = None
        self._init_registry()

    def _init_registry(self) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        bootstrap_default_model(self.models_dir)

        if not self.registry_path.exists():
            initial = {
                "models": {
                    self.default_model_name: {
                        "display_name": "Tiny Iris Logistic Regression",
                        "task_type": "tabular",
                        "format": ".joblib",
                        "path": str(self.models_dir / self.default_model_name),
                        "runtime_supported": True,
                        "source": "bootstrap",
                    }
                },
                "active_model_name": self.default_model_name,
            }
            self._write_registry(initial)

        state = self._read_registry()
        self.active_model_name = state.get("active_model_name", self.default_model_name)
        self._ensure_active_model_loaded()

    def _read_registry(self) -> Dict[str, Any]:
        return json.loads(self.registry_path.read_text())

    def _write_registry(self, payload: Dict[str, Any]) -> None:
        self.registry_path.write_text(json.dumps(payload, indent=2))

    def list_models(self) -> Dict[str, Any]:
        state = self._read_registry()
        return {
            "active_model_name": state.get("active_model_name"),
            "models": state.get("models", {}),
        }

    def register_model(self, filename: str, task_type: str, display_name: Optional[str] = None) -> Dict[str, Any]:
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_ACCEPTED_SUFFIXES:
            raise ValueError(f"Unsupported model suffix: {suffix}")

        runtime_supported = suffix in SUPPORTED_EXECUTABLE_TABULAR_SUFFIXES and task_type == "tabular"
        state = self._read_registry()
        state.setdefault("models", {})[filename] = {
            "display_name": display_name or filename,
            "task_type": task_type,
            "format": suffix,
            "path": str(self.models_dir / filename),
            "runtime_supported": runtime_supported,
            "source": "upload",
        }
        self._write_registry(state)
        return state["models"][filename]

    def select_model(self, model_name: str) -> Dict[str, Any]:
        with self._lock:
            state = self._read_registry()
            models = state.get("models", {})
            if model_name not in models:
                raise KeyError(f"Model not found: {model_name}")

            model_meta = models[model_name]
            if not model_meta.get("runtime_supported"):
                raise ValueError(f"Model is registered but not executable by current runtime: {model_name}")

            state["active_model_name"] = model_name
            self._write_registry(state)
            self.active_model_name = model_name
            self._loaded_models.pop(model_name, None)
            model = self.load_model(model_name)
            return {
                "active_model_name": model_name,
                "model_type": type(model).__name__,
                "metadata": model_meta,
            }

    def _ensure_active_model_loaded(self) -> None:
        if self.active_model_name:
            self.load_model(self.active_model_name)

    def load_model(self, model_name: str):
        with self._lock:
            if model_name in self._loaded_models:
                return self._loaded_models[model_name]

            state = self._read_registry()
            model_meta = state.get("models", {}).get(model_name)
            if not model_meta:
                raise KeyError(f"Model not found: {model_name}")

            # When not using file-backed models, return a dummy model for predictions
            if not self.use_file_models:
                model = DummyTabularModel()
                self._loaded_models[model_name] = model
                return model

            path = Path(model_meta["path"])
            suffix = path.suffix.lower()
            if suffix not in SUPPORTED_EXECUTABLE_TABULAR_SUFFIXES:
                raise ValueError(f"Current runtime cannot execute model format: {suffix}")
            if not path.exists():
                raise FileNotFoundError(f"Model file does not exist: {path}")

            model = joblib.load(path)
            self._loaded_models[model_name] = model
            return model

    def predict_tabular(self, features):
        if not self.active_model_name:
            raise RuntimeError("No active model selected")
        model = self.load_model(self.active_model_name)
        prediction = model.predict([features])[0]
        probabilities = getattr(model, "predict_proba", None)
        prob_values = None
        if probabilities:
            prob = probabilities([features])[0]
            if hasattr(prob, "tolist"):
                prob_values = prob.tolist()
            else:
                try:
                    prob_values = list(prob)
                except Exception:
                    prob_values = None
        return {
            "model_name": self.active_model_name,
            "prediction": int(prediction) if hasattr(prediction, "item") else prediction,
            "probabilities": prob_values,
        }

    def explain_active_model(self):
        if not self.active_model_name:
            raise RuntimeError("No active model selected")
        model = self.load_model(self.active_model_name)
        coefficients = getattr(model, "coef_", None)
        intercept = getattr(model, "intercept_", None)
        return {
            "model_name": self.active_model_name,
            "model_class": type(model).__name__,
            "coefficients": coefficients.tolist() if coefficients is not None else None,
            "intercept": intercept.tolist() if intercept is not None else None,
            "note": "This is a lightweight placeholder XAI response for the startup model.",
        }


def build_registry_from_env() -> ModelRegistry:
    models_dir = os.getenv("MODELS_DIR", "/app/models")
    registry_path = os.getenv("MODEL_REGISTRY_PATH", "/app/models/model_registry.json")
    default_model_name = os.getenv("ACTIVE_MODEL_NAME", "tiny-iris-logreg.joblib")
    use_file_models = os.getenv("USE_FILE_MODELS", "true").strip().lower() in {"1", "true", "yes", "on"}
    return ModelRegistry(
        models_dir=models_dir,
        registry_path=registry_path,
        default_model_name=default_model_name,
        use_file_models=use_file_models,
    )
