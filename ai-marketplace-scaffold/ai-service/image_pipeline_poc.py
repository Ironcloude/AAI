import os
from io import BytesIO
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.models import EfficientNet_V2_S_Weights, get_model

# Single source of truth for device
DEVICE = torch.device(os.getenv("TORCH_DEVICE", "cpu"))


def load_model(
    use_file: bool = True, path: Optional[str] = None
) -> Optional[Any]:
    """Load an image model or return None for heuristic mode."""
    if not use_file:
        return None

    if not path or not os.path.exists(path):
        return None

    # Use the same architecture loading logic as inference.py.
    # NOTE: If the model was trained as MTL (using MultiTaskClassifier),
    # define MultiTaskClassifier here or import it for torch.load to work.
    try:
        base_model = get_model("efficientnet_v2_s", weights=None)
        num_features = base_model.classifier[1].in_features
        base_model.classifier[1] = nn.Linear(num_features, 2)
        model = base_model

        # weights_only=False supports potential custom wrappers/classes
        state_dict = torch.load(path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}", flush=True)
        raise e


def _normalise_score(value: float) -> float:
    """Clamp a score to the range [0, 100] and round to 2 decimal places."""
    return round(max(0.0, min(100.0, value)), 2)


def _grade(color: float, size: float, ripeness: float) -> str:
    """Return a quality grade (A, B, or C) based on metric thresholds."""
    if color >= 75 and size >= 80 and ripeness >= 70:
        return "A"
    if color >= 65 and size >= 70 and ripeness >= 60:
        return "B"
    return "C"


def analyse_image(image_bytes: bytes, model: Optional[Any] = None) -> Dict:
    """Analyse image using a loaded model if available, else heuristics."""
    load_note = None
    if model is None:
        use_file = os.getenv("USE_FILE_MODELS", "true").strip().lower() in {
            "1", "true", "yes", "on"}
        image_model_path = os.getenv("IMAGE_MODEL_PATH")
        try:
            model = load_model(use_file=use_file, path=image_model_path)
            if model is None and use_file:
                load_note = (
                    "File-backed image model requested but not loaded; "
                    "falling back to heuristic."
                )
        except Exception as exc:
            model = None
            load_note = f"Image model load failed: {exc} — using heuristic."

    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    if model is not None:
        model.to(DEVICE)
        preprocess = EfficientNet_V2_S_Weights.DEFAULT.transforms()
        input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, class_idx = torch.max(probabilities, dim=1)

        classes = ["Healthy", "Rotten"]
        prediction = classes[class_idx.item()]
        confidence = float(confidence.item())

        # Calculate ripeness based on health confidence.
        # Healthy => confidence = ripeness; Rotten => 100 - confidence
        ripeness_score = (
            confidence * 100 if prediction == "Healthy"
            else (1 - confidence) * 100
        )

        return {
            "prediction": prediction,
            "confidence": confidence,
            "metrics": {
                "ripeness": round(ripeness_score, 2),
            },
            "explanation": {"method": "efficientnet_v2_s"}
        }

    # Fallback to heuristic
    arr = np.asarray(image).astype("float32")
    h, w, _ = arr.shape

    brightness = float(arr.mean()) / 255.0
    color_spread = float(arr.std()) / 128.0
    size_score = min((h * w) / (512 * 512), 1.0) * 100.0

    color_score = _normalise_score(65 + (color_spread * 35))
    ripeness_score = _normalise_score(55 + (brightness * 45))
    size_score = _normalise_score(size_score)
    overall_grade = _grade(color_score, size_score, ripeness_score)

    note = (
        load_note
        or "This is a runnable placeholder for Task 2 "
        "until you plug in a trained image model."
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
