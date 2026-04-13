from io import BytesIO
from typing import Dict, Optional, Any
import os

import numpy as np
from PIL import Image


def load_model(use_file: bool = True, path: Optional[str] = None) -> Optional[Any]:
    """Load an image model or return None for heuristic mode.

    - If use_file is False: return None (use heuristics).
    - If use_file is True and path is given: this is where you'd load your model (e.g., torch.load(...)).
      For this scaffold we keep it unimplemented and still return None.
    """
    if not use_file:
        return None
    # Placeholder: plug in real model loader here (torch, onnxruntime, etc.)
    # Example:
    # import torch
    # model = torch.jit.load(path)
    # model.eval()
    # return model
    return None


def _normalise_score(value: float) -> float:
    return round(max(0.0, min(100.0, value)), 2)


def _grade(color: float, size: float, ripeness: float) -> str:
    if color >= 75 and size >= 80 and ripeness >= 70:
        return "A"
    if color >= 65 and size >= 70 and ripeness >= 60:
        return "B"
    return "C"


def analyse_image(image_bytes: bytes, model: Optional[Any] = None) -> Dict:
    """Analyse image using a loaded model if available, else heuristics.

    - Uses USE_FILE_MODELS env to decide whether to try file-backed model loading.
    - An optional IMAGE_MODEL_PATH can be provided to point at the model file.
    """
    # Decide loading policy from env when caller hasn't provided a model
    load_note = None
    if model is None:
        use_file = os.getenv("USE_FILE_MODELS", "true").strip().lower() in {"1", "true", "yes", "on"}
        image_model_path = os.getenv("IMAGE_MODEL_PATH")
        try:
            model = load_model(use_file=use_file, path=image_model_path)
            if model is None and use_file:
                load_note = "File-backed image model requested but not loaded; falling back to heuristic."
        except Exception as exc:
            model = None
            load_note = f"Image model load failed: {exc} — using heuristic."

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    arr = np.asarray(image).astype("float32")
    h, w, _ = arr.shape

    if model is not None:
        # Example stub: adapt to your model API
        # outputs = model.predict(arr)  # not implemented in scaffold
        pass  # fall through to heuristic for now

    brightness = float(arr.mean()) / 255.0
    color_spread = float(arr.std()) / 128.0
    size_score = min((h * w) / (512 * 512), 1.0) * 100.0

    color_score = _normalise_score(65 + (color_spread * 35))
    ripeness_score = _normalise_score(55 + (brightness * 45))
    size_score = _normalise_score(size_score)
    overall_grade = _grade(color_score, size_score, ripeness_score)

    note = (
        load_note or "This is a runnable placeholder for Task 2 until you plug in a trained image model."
    )

    return {
        "metrics": {
            "color": color_score,
            "size": size_score,
            "ripeness": ripeness_score,
        },
        "overall_grade": overall_grade,
        "explanation": {
            "method": "heuristic-image-rubric",
            "note": note,
            "image_size": {"width": w, "height": h},
        },
    }
