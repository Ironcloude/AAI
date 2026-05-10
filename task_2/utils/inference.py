import sys
import cv2
from dataclasses import dataclass
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
sys.path.append("..")
sys.path.append(".")
from utils.generate_masks import generate_produce_mask
from utils.grade_produce import grade_colour, grade_colour_generic, grade_proportion, grade_colour_dual
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


@dataclass
class InferenceContext:
    model: nn.Module
    device: torch.device
    auto_transforms: object
    is_mtl: bool
    health_names: list
    type_names: list
    type_confidence_threshold: float = 50.0


def classify_produce(image_path, ctx: InferenceContext):
    """
    Classify a single image of produce.
    Returns the predicted health label and confidence.
    If the model is MTL, fruit type and type confidence is additionally returned.
    """
    image = Image.open(image_path).convert("RGBA").convert("RGB")
    input_tensor = ctx.auto_transforms(image).unsqueeze(0).to(ctx.device)

    with torch.no_grad():
        if ctx.is_mtl:
            health_output, type_output = ctx.model(input_tensor)
        else:
            health_output = ctx.model(input_tensor)

        health_probs = torch.nn.functional.softmax(health_output, dim=1)[0]
        health_pred_id = health_probs.argmax().item()
        health_confidence = health_probs[health_pred_id].item() * 100

    health_label = ctx.health_names[health_pred_id]
    print(f"Health: {health_label} ({health_confidence:.1f}%)")

    result = {
        "health_label": health_label,
        "health_confidence": health_confidence,
        "health_pred_id": health_pred_id,
        "input_tensor": input_tensor,
        "fruit_type": None,
        "type_confidence": None,
        "known_type": None,
        "type_pred_id": None,
    }

    if ctx.is_mtl:
        type_probs = torch.nn.functional.softmax(type_output, dim=1)[0]
        type_pred_id = type_probs.argmax().item()
        type_confidence = type_probs[type_pred_id].item() * 100

        fruit_type = ctx.type_names[type_pred_id]
        known = type_confidence >= ctx.type_confidence_threshold
        result["fruit_type"] = fruit_type if known else "unknown"
        result["type_confidence"] = type_confidence
        result["known_type"] = known

        print((f"Type:   {result['fruit_type']} ({type_confidence:.1f}%)"
               f"{'' if known else f' ({fruit_type} - {ctx.type_confidence_threshold - type_confidence:.1f} below threshold)'}"))

    return result


def grade_produce(image_path, ctx: InferenceContext, healthy_colour_refs=None,
                  rotten_colour_references=None, colour_distribution=None,
                  grade_colour_vs_ref=False, preview_img=True, composite_result=True,
                  softmax_g=0, cam=None) -> dict:
    """
    Determine grading based on ripeness, colour and proportion.
    Ripeness    - Softmax prediction confidence from classify_produce()
    Colour      - STL/fallback: Score of saturation/vibrancy (OpenCV)
                - MTL: HSV histogram similarity vs known produce healthy reference (OpenCV)
    Proportion  - Contour solidity (OpenCV)
    """
    if preview_img and cam is None:
        raise ValueError("cam must be provided when preview_img=True")

    mask = generate_produce_mask(str(image_path), method="rembg")
    classification = classify_produce(image_path, ctx)

    label = classification["health_label"]
    fruit_type = classification["fruit_type"]

    ripeness_score = (
        classification["health_confidence"]
        if label == "Healthy"
        else 100 - classification["health_confidence"]
    )

    generic_score, vibrancy = grade_colour_generic(image_path, colour_distribution, mask=mask)
    if classification.get("known_type") and rotten_colour_references and grade_colour_vs_ref:
        dual_score, summary_str = grade_colour_dual(image_path, fruit_type,
                                                    healthy_colour_refs, rotten_colour_references,
                                                    mask=mask, gamma=softmax_g)
        colour_score = max(dual_score, generic_score)
    elif classification.get("known_type") and healthy_colour_refs and grade_colour_vs_ref:
        colour_score, summary_str = grade_colour(image_path, fruit_type,
                                                 healthy_colour_refs, mask=mask), ""
    else:
        colour_score, summary_str = generic_score, ""
    print(summary_str)
    proportion_score = grade_proportion(image_path, mask=mask)

    scores = {
        "ripeness": ripeness_score,
        "colour": colour_score,
        "size": proportion_score,
    }

    THRESHOLDS = [
        ("A", {"colour": 85, "size": 90, "ripeness": 80}),
        ("B", {"colour": 75, "size": 80, "ripeness": 70}),
        ("C", {"colour": 65, "size": 70, "ripeness": 60}),
    ]
    COMPOSITE_THRESHOLDS = [
        (grade, sum(criteria.values()) / len(criteria))
        for grade, criteria in THRESHOLDS
    ]
    composite_score = 0

    if composite_result:
        composite_score = (ripeness_score + colour_score + proportion_score) / 3
        grade = "D"
        for grade_name, avg_threshold in COMPOSITE_THRESHOLDS:
            if composite_score >= avg_threshold:
                grade = grade_name
                break
    else:
        grade = "D"
        for threshold_grade, minimums in THRESHOLDS:
            if all(scores[key] >= minimums[key] for key in minimums):
                grade = threshold_grade
                break

    if label == "Rotten":
        grade = "Rejected"

    used_ref = classification.get("known_type") and healthy_colour_refs and grade_colour_vs_ref
    if used_ref:
        colour_method = f"ref-similarity, generic: {int(generic_score)}%"
    elif classification.get("known_type"):
        colour_method = "vibrancy percentile"
    else:
        colour_method = "vibrancy percentile (type unknown)"

    LBL = 13
    VAL = 4
    if composite_result:
        header = f"Composite grade: {grade}  ({composite_score:.1f}/100)"
    else:
        header = f"Grade: {grade}"

    explanation = "\n".join([
        header,
        f"{'Ripeness:':<{LBL}}{int(ripeness_score):>{VAL-1}}%   (health confidence)",
        f"{'Colour:':<{LBL}}{int(colour_score):>{VAL-1}}%   ({colour_method})",
        f"{'Proportion:':<{LBL}}{int(proportion_score):>{VAL-1}}%   (contour solidity)",
    ])

    if preview_img:
        image_tensor = classification["input_tensor"]
        predicted_health_idx = classification["health_pred_id"]
        targets = [ClassifierOutputTarget(predicted_health_idx)]
        grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0, :]

        mean = torch.tensor([0.485, 0.456, 0.406], device=image_tensor.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=image_tensor.device).view(1, 3, 1, 1)
        denorm = (image_tensor * std + mean).clamp(0, 1)
        rgb_img = denorm[0].detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)

        cam_visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        img = Image.open(image_path).convert('RGB')
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].imshow(img)
        axes[0].set_title("Original")
        axes[0].axis("off")

        mask_display = cv2.resize(
            mask, (img.width, img.height),
            interpolation=cv2.INTER_NEAREST) if mask.shape[:2] != (img.height, img.width) else mask
        axes[1].imshow(mask_display, cmap="gray")
        axes[1].set_title("Segmentation Mask")
        axes[1].axis("off")

        axes[2].imshow(cam_visualization)
        plt.title(f"Grad-CAM (Health: {ctx.health_names[predicted_health_idx]})")
        axes[2].axis("off")
        plt.tight_layout()

        plt.figtext(0.5, 0.02, f"{explanation}\n{summary_str}", ha="center", va="top",
                    fontsize=11, family="monospace", multialignment="left",
                    bbox={"facecolor": "white", "alpha": 0.85, "pad": 8, "edgecolor": "#444", "boxstyle": "round,pad=0.6"})
        plt.show()

    return {
        "prediction": label,
        "fruit_type": fruit_type,
        "ripeness_score": ripeness_score,
        "color_score": colour_score,
        "generic_colour_score": generic_score,
        "proportion_score": proportion_score,
        "overall_grade": grade,
        "explanation": explanation,
    }
