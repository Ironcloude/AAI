"""Segmentation mask generation using SAM 2, rembg, or GrabCut."""

import gc
from pathlib import Path
import numpy as np
import cv2
import torch
from rembg import new_session, remove

def _segment_sam2(image: np.ndarray, sam2_predictor: "SAM2ImagePredictor"
                  ) -> np.ndarray:
    """Segment produce using SAM 2 with an inner-80% box prompt.

    Assumes the produce is roughly centered in the image.

    Args:
        image: BGR image array.
        sam2_predictor: Initialised SAM 2 image predictor.

    Returns:
        Binary mask with values 0 (background) and 1 (foreground).
    """
    height, width = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    sam2_predictor.set_image(image_rgb)

    # Targets the inner 80%
    margin_x, margin_y = int(width * 0.1), int(height * 0.1)
    bounding_box = np.array([margin_x, margin_y, 
                             width - margin_x, height - margin_y])
    # Returns 3 different masks as different granularity
    masks, scores, _ = sam2_predictor.predict(
        box=bounding_box,
        multimask_output=True,
    )

    # Pick the mask with highest confidence
    best_mask = masks[scores.argmax()]
    return best_mask.astype(np.uint8)

def _segment_rembg(image: np.ndarray, session=None) -> np.ndarray:
    """Segment using rembg (U²-Net salient object detection).

    Args:
        image: BGR image array.
        session: Optional pre-built rembg session for batch efficiency.

    Returns:
        Binary mask with values 0 (background) and 1 (foreground).
    """
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = remove(img_rgb, only_mask=True, session=session)
    return (result > 127).astype(np.uint8)

def _segment_grabcut(image: np.ndarray) -> np.ndarray:
    """Segment using OpenCV's GrabCut with an inner-80% box prompt.

    Assumes the produce is roughly centered in the image.

    Args:
        image: BGR image array.

    Returns:
        Binary mask with values 0 (background) and 1 (foreground).
    """
    height, width = image.shape[:2]
    mask = np.zeros((height, width), np.uint8)

    # Targets the inner 80%
    margin_x, margin_y = int(width * 0.1), int(height * 0.1)
    bounding_box = (margin_x, margin_y, width - 2 * margin_x, height - 2 * margin_y)

    # Populates mask with pixel labels :
    # (definite fg, definite bg, probable fg, probable bg )
    cv2.grabCut(image, mask, bounding_box, 
                bg_model=np.zeros((1, 65), np.float64), # Boilerplate
                fg_model=np.zeros((1, 65), np.float64), 
                iterCount=5, mode=cv2.GC_INIT_WITH_RECT
                )

    # Mark definite (GC_FGD) + probable foreground (CD_PR_FGD) as 1, rest as 0
    return np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)

def generate_produce_mask(data_path: str | Path | None = None,
                          output_dir: str | Path | None = None, 
                          method: str = "rembg", sam_model_size: str = "small",
                          device: str = "cuda" ) -> np.ndarray | None:
    """Generate segmentation masks for a single image or a whole dataset.

    In dataset mode, mirrors the dataset folder structure under'output_dir' 
    and skips existing masks for checkpointing.
    Masks are generated on original (unaugmented) images.

    Args:
        data_path: Path to dataset directory or an individual image.
        output_dir: Where to save masks. 
                    Defaults to `<data_path>/../<method>_masks/`.
        method: "grabcut", "rembg", or "sam2".
        sam_model_size: "small" or "large" (SAM 2 only).
        device: "cuda" or "cpu" (SAM 2 only).

    Returns:
        In single-image mode, the binary mask as a numpy array.
        In dataset mode, None (masks are written to disk).
    """
    data_path = Path(data_path)

    # Load segmentation method
    if method == "sam2":
        from sam2.build_sam import build_sam2_hf
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        print(f"Loading SAM 2 ({sam_model_size}) on {device}...")
        model_id = f"facebook/sam2.1-hiera-{sam_model_size}"
        sam_predictor = SAM2ImagePredictor(build_sam2_hf(model_id, device=device))

    # Single image mode
    if data_path.is_file():
        image = cv2.imread(str(data_path))
        if image is None:
            raise Exception(f"Could not read image: {data_path}")
        if method == "sam2":
            return _segment_sam2(image, sam_predictor)
        elif method == "rembg":
            return _segment_rembg(image)
        else:
            return _segment_grabcut(image)
        
    # Dataset mode 
    if output_dir is None:
        output_dir = data_path.parent / f"{method}{'_'+ sam_model_size if method == 'sam2' else ''}_masks"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total, skipped = 0, 0
    rembg_session = new_session("u2net") # Create session for batch processing
    for class_dir in sorted(data_path.iterdir()):
        if not class_dir.is_dir():
            continue
        # Skip rotten samples
        if "healthy" not in class_dir.name.lower():
            continue

        mask_class_dir = output_dir / class_dir.name
        mask_class_dir.mkdir(parents=True, exist_ok=True)
        class_total, class_skipped = 0, 0

        for img_path in sorted(class_dir.iterdir()):
            mask_path = mask_class_dir / (img_path.stem + ".png")

            if mask_path.exists():
                skipped += 1
                class_skipped += 1
                continue

            image = cv2.imread(str(img_path))
            if image is None:
                continue

            if method == "sam2":
                binary_mask = _segment_sam2(image, sam_predictor)
            elif method == "rembg":

                binary_mask = _segment_rembg(image, session=rembg_session)
            else:
                binary_mask = _segment_grabcut(image)

            cv2.imwrite(str(mask_path), binary_mask * 255)
            total += 1
            class_total += 1
            if total % 50 == 0:
                print(f"  {total} masks generated so far...")

        print(f"{class_dir.name}: {class_total} generated, {class_skipped} cached")

    print(f"\nDone. {total} masks generated, {skipped} skipped (already cached).")
    print(f"Saved to: {output_dir}")

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()

    script_dir = Path(__file__).resolve().parent
    dataset_dir = script_dir.parent / "data" / "Fruit_And_Vegetable_Diseases_Dataset"
    method = "rembg"
    generate_produce_mask(data_path=dataset_dir, method=method)