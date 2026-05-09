"""Segmentation mask generation using SAM 2, rembg, or GrabCut."""

import gc
from pathlib import Path
import numpy as np
import cv2
import torch
from sam2.build_sam import build_sam2_hf
from sam2.sam2_image_predictor import SAM2ImagePredictor
from rembg import new_session, remove

def _segment_sam2(image: np.ndarray, sam2_predictor: SAM2ImagePredictor
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
        # Use HF implementation for convenience
        # https://github.com/facebookresearch/sam2/blob/main/sam2/build_sam.py
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
        # if "healthy" not in class_dir.name.lower():
            # continue

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

def visualize_segmentation(image_path: str | Path,
                           method: str = "sam2",
                           sam_predictor: SAM2ImagePredictor | None = None,
                           sam_model_size: str = "small",
                           device: str = "cuda",
                           save_path: str | Path | None = None) -> tuple:
    """Three-panel figure (original | mask | segmented) for a single image.

    Mirrors the rembg-style report figure for any segmentation method, so SAM 2
    and GrabCut can be visualised consistently. Caller can pass a pre-loaded
    `sam_predictor` to avoid the ~1GB model reload between visualisations.

    Args:
        image_path: Path to the source image.
        method: "sam2", "rembg", or "grabcut".
        sam_predictor: Optional pre-loaded SAM 2 predictor (used only when
            `method == "sam2"`); built on demand if not supplied.
        sam_model_size: SAM 2 size ("small" or "large").
        device: SAM 2 device.
        save_path: Optional output path for the figure (PNG recommended).

    Returns:
        Tuple of (matplotlib figure, binary mask in {0, 1}).
    """
    import matplotlib.pyplot as plt  # local import — keep module light

    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read {image_path}")

    if method == "sam2":
        if sam_predictor is None:
            print(f"Loading SAM 2 ({sam_model_size}) on {device}...")
            model_id = f"facebook/sam2.1-hiera-{sam_model_size}"
            sam_predictor = SAM2ImagePredictor(build_sam2_hf(model_id, device=device))
        mask = _segment_sam2(image, sam_predictor)
    elif method == "rembg":
        mask = _segment_rembg(image)
    elif method == "grabcut":
        mask = _segment_grabcut(image)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'sam2', 'rembg', or 'grabcut'.")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    segmented = image_rgb.copy()
    segmented[mask == 0] = 0  # black-out background

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"Segmentation: {method}", fontsize=12)

    axes[0].imshow(image_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Mask")
    axes[1].axis("off")

    axes[2].imshow(segmented)
    axes[2].set_title("Segmented")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"Saved {save_path}")
    return fig, mask


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    sample  = "/home/jaime/AAI/AAI/task_2/data/Fruit_And_Vegetable_Diseases_Dataset_no_identical_no_aug/Orange__Healthy/freshOrange (3).png"
    script_dir = Path(__file__).resolve().parent
    dataset_dir = script_dir.parent / "data" / "Fruit_And_Vegetable_Diseases_Dataset_no_identical_no_aug"
    if not sample: 
        method = "rembg"
        generate_produce_mask(data_path=dataset_dir, method=method)
    # SAM 2 — same image you used for the rembg figure
    fig, mask = visualize_segmentation(
        sample,
        method="sam2",
        sam_model_size="small",
        save_path=f"task_2/figures/{sample.basename}"
    )

    # rembg — same call signature, just swap method
    fig, mask = visualize_segmentation(
        sample,
        method="rembg",
        save_path=f"task_2/figures/{sample.basename}"
    )

    # # grabcut — same again
    # fig, mask = visualize_segmentation(
    #     sample,
    #     method="grabcut",
    #     save_path="figures/seg_grabcut_cucumber.png"
    # )
