"""Produce quality grading pipeline using trained MTL/STL vision model.

Architecture and STL/MTL mode are detected from the EX-prefix in the model filename.
Colour grading uses histogram similarity (known type) or generic percentile scoring.
Proportion grading uses contour solidity via rembg segmentation.
Falls back to a pixel-level heuristic when no model is available.
"""

import os
import pickle
import re
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import load_file
from torchvision.models import get_model, get_weight
from utils.grade_produce import grade_colour_generic, grade_proportion
from utils.generate_masks import generate_produce_mask
from torchvision.models import get_model_weights
from utils.mtl_model import MultiTaskClassifier

    
DEVICE = torch.device(os.getenv("TORCH_DEVICE", "cpu"))
HEALTH_NAMES = ["Healthy", "Rotten"]
TYPE_CONFIDENCE_THRESHOLD = 50

_GRADE_THRESHOLDS = [
    ("A", {"colour": 85, "size": 90, "ripeness": 80}),
    ("B", {"colour": 75, "size": 80, "ripeness": 70}),
    ("C", {"colour": 65, "size": 70, "ripeness": 60}),
]

# Cached colour references - loaded once on first inference call
_colour_refs: Optional[dict] = None
_colour_dist: Optional[dict] = None
_refs_loaded = False


def _load_colour_refs() -> tuple:
    global _colour_refs, _colour_dist, _refs_loaded
    if _refs_loaded:
        return _colour_refs, _colour_dist
    _refs_loaded = True
    # Search MODELS_DIR first (volume mount), then the task_2 data mount
    for base in [Path(os.getenv("MODELS_DIR", "/app/models")), Path("/aai_service/task_2/data")]:
        ref  = base / "colour_references_rembg.pkl"
        dist = base / "generic_colour_distribution.pkl"
        if ref.exists() and dist.exists():
            with open(ref, "rb") as f:
                _colour_refs = {k.lower(): v for k, v in pickle.load(f).items()}
            with open(dist, "rb") as f:
                _colour_dist = {k.lower(): v for k, v in pickle.load(f).items()}
            print(f"[image_pipeline] Colour refs loaded from {base}", flush=True)
            return _colour_refs, _colour_dist
    print("[image_pipeline] Colour reference files not found; falling back to generic heuristic.", flush=True)
    return None, None


def _detect_experiment(stem: str):
    """Look up the experiment config from experiment_configs matching the EX-prefix in the filename."""
    m = re.search(r"\b(EX\d+)\b", stem, re.IGNORECASE)
    if not m:
        return None
    prefix = m.group(1).upper()
    from experiment_configs import task_2_config as experiments
    for name, obj in vars(experiments).items():
        if hasattr(obj, "architecture") and name.upper().startswith(prefix + "_"):
            return obj
    return None


def load_model(use_file: bool = True, path: Optional[str] = None,
               config: Optional[dict] = None) -> Optional[Any]:
    """Load a trained image model from .pth or .safetensors.

    Config (from registry metadata set at upload time) takes priority.
    Falls back to EX-prefix detection via experiment_configs for legacy filenames.
    """
    if not use_file or not path or not os.path.exists(path):
        return None

    model_path = Path(path)

    if config:
        arch = config["architecture"]
        is_mtl = config.get("is_mtl", False)
        num_produce_types = config.get("num_produce_types", 36)
        weights = get_model_weights(arch).DEFAULT
    else:
        exp = _detect_experiment(model_path.stem)
        num_produce_types = int(os.getenv("NUM_PRODUCE_TYPES", "36"))
        if exp:
            arch, is_mtl = exp.architecture, exp.is_mtl
            weights = get_weight(exp.weight_string)
        else:
            arch, is_mtl = "efficientnet_v2_s", False
            weights = get_weight("EfficientNet_V2_S_Weights.IMAGENET1K_V1")
            print(f"[image_pipeline] No config for '{model_path.name}'; defaulting to {arch} STL.", flush=True)
    base_model = get_model(arch, weights=None)

    if is_mtl:
        model = MultiTaskClassifier(base_model, num_produce_classes=num_produce_types)
    else:
        if hasattr(base_model, "classifier") and isinstance(base_model.classifier, nn.Sequential):
            in_features = base_model.classifier[-1].in_features
            base_model.classifier[-1] = nn.Linear(in_features, 2)
        elif hasattr(base_model, "head") and isinstance(base_model.head, nn.Linear):
            in_features = base_model.head.in_features
            base_model.head = nn.Linear(in_features, 2)
        model = base_model

    suffix = model_path.suffix.lower()
    if suffix == ".safetensors":
        state_dict = load_file(str(model_path))
    else:
        state_dict = torch.load(str(model_path), map_location=DEVICE, weights_only=True)

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    model._is_mtl = is_mtl
    model._transforms = weights.transforms()
    return model


def _classify(image_pil: Image.Image, model: Any) -> dict:
    transforms = getattr(model, "_transforms", None)
    if transforms is None:
        raise ValueError("Model has no _transforms — must be loaded via load_model()")
    input_tensor = transforms(image_pil).unsqueeze(0).to(DEVICE)

    is_mtl = getattr(model, "_is_mtl", False)
    with torch.no_grad():
        out = model(input_tensor)
        health_out = out[0] if is_mtl else out

    health_probs = torch.nn.functional.softmax(health_out, dim=1)[0]
    pred_id = health_probs.argmax().item()
    label = HEALTH_NAMES[pred_id]
    confidence = float(health_probs[pred_id].item() * 100)

    fruit_type, known_type = None, False
    if is_mtl:
        type_probs = torch.nn.functional.softmax(out[1], dim=1)[0]
        type_pred_id = type_probs.argmax().item()
        type_confidence = float(type_probs[type_pred_id].item() * 100)
        known_type = type_confidence >= TYPE_CONFIDENCE_THRESHOLD
        fruit_type = str(type_pred_id) if known_type else "unknown"

    return {
        "health_label": label,
        "health_confidence": confidence,
        "fruit_type": fruit_type,
        "known_type": known_type,
    }


def analyse_image(image_bytes: bytes, model: Optional[Any] = None) -> Dict:
    """Grade produce quality from raw image bytes."""
    if model is None:
        use_file = os.getenv("USE_FILE_MODELS", "true").strip().lower() in {"1", "true", "yes", "on"}
        try:
            model = load_model(use_file=use_file, path=os.getenv("IMAGE_MODEL_PATH"))
        except Exception as exc:
            return _heuristic_fallback(image_bytes, note=f"Model load failed: {exc}")

    if model is None:
        return _heuristic_fallback(image_bytes, note="No image model loaded.")


    image_pil = Image.open(BytesIO(image_bytes)).convert("RGBA").convert("RGB")
    _, colour_dist = _load_colour_refs()

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        image_pil.save(tmp_path, format="JPEG")

    try:
        clf = _classify(image_pil, model)
        label = clf["health_label"]
        fruit_type = clf["fruit_type"]

        ripeness = clf["health_confidence"] if label == "Healthy" else 100 - clf["health_confidence"]
        mask = generate_produce_mask(str(tmp_path), method="rembg")

        if colour_dist is not None:
            generic_score, vibrancy, uniformity = grade_colour_generic(tmp_path, colour_dist, mask=mask)
        else:
            generic_score, vibrancy, uniformity = 50.0, 0.0, 0.0

        # fruit_type is a numeric class index string — histogram refs require the
        # produce name (e.g. "apple_healthy"). Always use generic scoring until a
        # class index→name mapping is available in the service.
        colour_score = generic_score

        size_score = grade_proportion(tmp_path, mask=mask)
    finally:
        tmp_path.unlink(missing_ok=True)

    scores = {"colour": colour_score, "size": size_score, "ripeness": ripeness}

    if label == "Rotten":
        overall_grade = "Rejected"
    else:
        overall_grade = "D"
        for grade, mins in _GRADE_THRESHOLDS:
            if all(scores[k] >= mins[k] for k in mins):
                overall_grade = grade
                break

    colour_detail = f"vibrancy: {vibrancy:.2f}, uniformity: {uniformity:.2f}"

    return {
        "prediction": label,
        "fruit_type": fruit_type or "unknown",
        "overall_grade": overall_grade,
        "metrics": {
            "ripeness": round(ripeness, 2),
            "colour":   round(colour_score, 2),
            "size":     round(size_score, 2),
        },
        "explanation": {
            "method": "mtl" if getattr(model, "_is_mtl", False) else "stl",
            "health_confidence": round(clf["health_confidence"], 2),
            "colour_detail": colour_detail,
        },
    }


def _heuristic_fallback(image_bytes: bytes, note: str = "") -> Dict:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    arr = np.asarray(image).astype("float32")
    h, w, _ = arr.shape
    brightness   = float(arr.mean()) / 255.0
    color_spread = float(arr.std()) / 128.0
    size_score   = min((h * w) / (512 * 512), 1.0) * 100.0
    colour_score = round(max(0.0, min(100.0, 65 + color_spread * 35)), 2)
    ripeness     = round(max(0.0, min(100.0, 55 + brightness * 45)), 2)
    size_score   = round(max(0.0, min(100.0, size_score)), 2)

    if colour_score >= 75 and size_score >= 80 and ripeness >= 70:
        overall_grade = "A"
    elif colour_score >= 65 and size_score >= 70 and ripeness >= 60:
        overall_grade = "B"
    else:
        overall_grade = "C"

    return {
        "prediction": "Unknown",
        "fruit_type": "unknown",
        "overall_grade": overall_grade,
        "metrics": {"colour": colour_score, "size": size_score, "ripeness": ripeness},
        "explanation": {
            "method": "heuristic-pixel-rubric",
            "note": note or "Heuristic fallback.",
            "image_size": {"width": w, "height": h},
        },
    }
