"""Produce quality grading pipeline using trained MTL/STL vision model.

Architecture and MTL mode are taken from the model registry config.
Colour grading uses histogram similarity (known type) or generic percentile scoring.
Proportion grading uses contour solidity via rembg segmentation.
Falls back to a pixel-level heuristic when no model is available.
"""

import base64
import os
import pickle
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
# from safetensors.torch import load_file
from torchvision.models import get_model
from utils.inference import InferenceContext, grade_produce
from utils.xai_wrapper import TaskWrapper, swin_reshape_transform
from torchvision.models import get_model_weights
from utils.mtl_model import MultiTaskClassifier
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

DEVICE = torch.device(os.getenv("TORCH_DEVICE", "cpu"))
HEALTH_NAMES = ["Healthy", "Rotten"]
TYPE_CONFIDENCE_THRESHOLD = 50


# Cached colour references - loaded once on first inference call
_colour_refs: Optional[dict] = None
_rotten_refs: Optional[dict] = None
_colour_dist: Optional[dict] = None
_type_names_sorted: Optional[list] = None  # sorted once after refs load
_refs_loaded = False

# Cached GradCAM wrapper - keyed by model object id to avoid stale refs
_grad_cam_cache: dict = {}  # {model_id: (cam_wrapper, cam)}


def _load_colour_refs() -> tuple:
    global _colour_refs, _rotten_refs, _colour_dist, _type_names_sorted, _refs_loaded
    if _refs_loaded:
        return _colour_refs, _colour_dist, _rotten_refs
    _refs_loaded = True
    for base in [Path(os.getenv("MODELS_DIR", "/app/models")), Path("/aai_service/task_2/data")]:
        ref        = base / "colour_references_rembg.pkl"
        dist       = base / "generic_colour_distribution.pkl"
        rotten_ref = base / "colour_references_rembg_rotten.pkl"
        if ref.exists() and dist.exists():
            with open(ref, "rb") as f:
                _colour_refs = {k.lower(): v for k, v in pickle.load(f).items()}
            with open(dist, "rb") as f:
                _colour_dist = {k.lower(): v for k, v in pickle.load(f).items()}
            _type_names_sorted = sorted(_colour_refs.keys())
            if rotten_ref.exists():
                with open(rotten_ref, "rb") as f:
                    _rotten_refs = {k.lower(): v for k, v in pickle.load(f).items()}
                print(f"[image_pipeline] Colour refs + rotten refs loaded from {base}", flush=True)
            else:
                print(f"[image_pipeline] Colour refs loaded from {base} (no rotten refs)", flush=True)
            return _colour_refs, _colour_dist, _rotten_refs
    print("[image_pipeline] Colour reference files not found; falling back to generic heuristic.", flush=True)
    return None, None, None


def load_model(use_file: bool = True, path: Optional[str] = None,
               config: Optional[dict] = None) -> Optional[Any]:
    """Load a trained image model from .pth or .safetensors using registry config."""
    if not use_file or not path or not os.path.exists(path):
        return None

    if not config:
        raise ValueError(f"No config provided for model '{path}'. All models must be registered with architecture metadata.")

    model_path = Path(path)
    arch = config["architecture"]
    is_mtl = config.get("is_mtl", False)
    num_produce_types = config.get("num_produce_types", 36)
    weights = get_model_weights(arch).DEFAULT
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
        from safetensors.torch import load_file
        state_dict = load_file(str(model_path))
    else:
        state_dict = torch.load(str(model_path), map_location=DEVICE, weights_only=True)

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    model._is_mtl = is_mtl
    model._transforms = weights.transforms()
    return model


def _make_context(model: Any) -> InferenceContext:
    return InferenceContext(
        model=model,
        device=DEVICE,
        auto_transforms=getattr(model, "_transforms", None),
        is_mtl=getattr(model, "_is_mtl", False),
        health_names=HEALTH_NAMES,
        type_names=_type_names_sorted or [],
        type_confidence_threshold=TYPE_CONFIDENCE_THRESHOLD,
    )

def _mask_to_b64(mask: np.ndarray) -> str:
    arr = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _grad_cam_config(backbone) -> tuple[list, Any]:
    """Return (target_layers, reshape_transform) for supported backbones, or (None, None)."""
    from torchvision.models import SwinTransformer, EfficientNet, MaxVit
    if isinstance(backbone, SwinTransformer):
        return [backbone.norm], swin_reshape_transform
    if isinstance(backbone, EfficientNet):
        return [backbone.features[-1]], None
    if isinstance(backbone, MaxVit):
        return [backbone.blocks[-1]], None
    return None, None


def _try_grad_cam(model: Any, input_tensor: Any, clf: dict,
                  original_image: Optional[Image.Image] = None) -> Optional[str]:
    return None
    try:
        if not getattr(model, "_is_mtl", False) or not hasattr(model, "backbone"):
            return None

        model_id = id(model)
        if model_id not in _grad_cam_cache:
            cam_wrapper = TaskWrapper(model, target_task="health").eval()
            target_layers, reshape_transform = _grad_cam_config(cam_wrapper.base_model.backbone)
            if target_layers is None:
                print(f"[image_pipeline] GradCAM not supported for {type(cam_wrapper.base_model.backbone).__name__}", flush=True)
                return None
            cam = GradCAM(model=cam_wrapper, target_layers=target_layers,
                          reshape_transform=reshape_transform)
            _grad_cam_cache[model_id] = (cam_wrapper, cam)
        _, cam = _grad_cam_cache[model_id]

        targets = [ClassifierOutputTarget(clf["health_pred_id"])]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

        if original_image is not None:
            # Resize CAM mask to original image dimensions and overlay on original
            orig_w, orig_h = original_image.size
            import cv2 as _cv2
            grayscale_cam_full = _cv2.resize(grayscale_cam, (orig_w, orig_h))
            rgb_img = np.array(original_image).astype(np.float32) / 255.0
        else:
            # Fallback: denormalise the model input tensor (224×224)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            denorm = (input_tensor * std + mean).clamp(0, 1)
            rgb_img = denorm[0].detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
            grayscale_cam_full = grayscale_cam

        cam_vis = show_cam_on_image(rgb_img, grayscale_cam_full, use_rgb=True)

        buf = BytesIO()
        Image.fromarray(cam_vis).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as exc:
        print(f"[image_pipeline] GradCAM skipped: {exc}", flush=True)
        return None


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


    # Open image
    image_pil = Image.open(BytesIO(image_bytes)).convert("RGBA").convert("RGB")
    # Load colour references
    colour_refs, colour_dist, rotten_refs = _load_colour_refs()

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        image_pil.save(tmp_path, format="JPEG")

    try:
        context = _make_context(model)
        result = grade_produce(
            tmp_path, context,
            healthy_colour_refs=colour_refs,
            rotten_colour_references=rotten_refs,
            colour_distribution=colour_dist,
            grade_colour_vs_ref=True,
            preview_img=False,
            softmax_g=5,
        )
        clf = result["classification"]
        mask = result["mask"]
        mask_b64 = _mask_to_b64(mask)
        grad_cam_b64 = _try_grad_cam(model, clf["input_tensor"], clf, original_image=image_pil)
        
    finally:
        tmp_path.unlink(missing_ok=True)

    type_name = result["fruit_type"] or "unknown"
    colour_score = result["color_score"]
    raw_dual_score = result.get("raw_dual_score")
    generic_score = result["generic_colour_score"]
    type_confidence = clf["type_confidence"]
    colour_details = result.get("colour_details")

    if colour_details:
        colour_breakdown = {
            "method": "dual_bhattacharyya",
            "type_name": type_name,
            "known": True,
            "type_confidence": round(type_confidence, 1) if type_confidence else None,
            "d_healthy": colour_details["d_H"],
            "d_rotten":  colour_details["d_R"],
            "w_healthy": colour_details["w_H"],
            "w_rotten":  colour_details["w_R"],
            "gamma":     colour_details["gamma"],
            "dual_score":    round(raw_dual_score, 2) if raw_dual_score is not None else round(colour_score, 2),
            "generic_score": round(generic_score, 2),
            "final_score":   round(colour_score, 2),
            "histogram_b64": colour_details.get("histogram_b64"),
        }
    else:
        colour_breakdown = {
            "method": "vibrancy_percentile",
            "type_name": type_name,
            "known": clf["known_type"],
            "type_confidence": round(type_confidence, 1) if type_confidence else None,
            "score": round(colour_score, 2),
        }

    return {
        "prediction": result["prediction"],
        "fruit_type": type_name,
        "overall_grade": result["overall_grade"],
        "metrics": {
            "ripeness": round(result["ripeness_score"], 2),
            "colour":   round(colour_score, 2),
            "size":     round(result["proportion_score"], 2),
        },
        "colour_breakdown": colour_breakdown,
        "explanation": {
            "method": "mtl" if context.is_mtl else "stl",
            "health_confidence": round(clf["health_confidence"], 2),
            "composite_score": round(result["composite_score"], 2),
            "known_type": clf["known_type"],
            "type_confidence": round(type_confidence, 1) if type_confidence else None,
        },
        "proportion_details": result.get("proportion_details"),
        "xai": {
            "mask_b64": mask_b64,
            "grad_cam_b64": grad_cam_b64,
            "proportion_overlay_b64": result.get("proportion_overlay"),
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
        "xai": None,
    }
