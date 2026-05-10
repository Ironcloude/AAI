"""Colour and proportion grading functions for produce quality assessment."""

import sys
from pathlib import Path
import cv2
from matplotlib.pyplot import hsv
import numpy as np
sys.path.append(".")
try:
    from utils.generate_masks import generate_produce_mask
except ModuleNotFoundError:
    from generate_masks import generate_produce_mask
import pickle
from pathlib import Path
    
def grade_colour_dual(image_path, fruit_type,
                     healthy_refs, rotten_refs,
                     mask=None, verbose=True, gamma=0 ) -> float | None:
    img = cv2.imread(str(image_path))
    max_dim = 512
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if mask is None:
        mask = generate_produce_mask(str(image_path))
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    fruit_key = fruit_type.split("__")[0].strip()
    if fruit_key not in healthy_refs or fruit_key not in rotten_refs:
        if verbose:
            print(f"'{fruit_key}' missing from one of the references, returning None")
        return None

    hist = cv2.calcHist([hsv], [0, 1], mask, [30, 32], [0, 180, 0, 256]).astype(np.float32)
    hist /= (hist.sum() + 1e-8)

    def _norm(ref):
        ref = ref.astype(np.float32)
        return ref / (ref.sum() + 1e-8)

    h_ref = _norm(healthy_refs[fruit_key]["mean"])
    r_ref = _norm(rotten_refs[fruit_key]["mean"])
    distance_healthy = cv2.compareHist(hist, h_ref, cv2.HISTCMP_BHATTACHARYYA)
    distance_rotten = cv2.compareHist(hist, r_ref, cv2.HISTCMP_BHATTACHARYYA)
    if distance_healthy + distance_rotten == 0: # Division by zero gaurd
        return None
    
    # for plot display — compact, aligned, separated from main scores by a rule
    summary_lines = [
        "─" * 38,
        f"Bhattacharyya: d_H={distance_healthy:.3f}  d_R={distance_rotten:.3f}",
    ]
    
    if verbose:
        print("\n" + "="*50)
        print(f"DUAL COLOUR GRADING: {fruit_key.upper()}")
        print("-" * 50)
        print("1. Raw Bhattacharyya Distances (Lower is better):")
        print(f"{'d_healthy':<10} = {distance_healthy:.4f}")
        print(f"{'d_rotten':<10} = {distance_rotten:.4f}")

    if gamma:
        # Exponential decay function
        weight_healthy = np.exp(-gamma * distance_healthy)
        weight_rotten = np.exp(-gamma * distance_rotten)
        
        score = (weight_healthy / (weight_healthy + weight_rotten)) * 100
        score_ref = (distance_rotten / (distance_healthy + distance_rotten)) * 100

        summary_lines.append(f"Weighted (γ={gamma}): w_H={weight_healthy:.3f}  w_R={weight_rotten:.3f}")
        summary_lines.append(f"Score: {score:.1f}%")
        
        if verbose:
            print(f"\n2. Exponential Weighting (gamma = {gamma}):")
            print(f"{'Formula:':<10} w = exp(-gamma * distance)")
            print(f"{'w_healthy':<10} = exp(-{gamma:<4} * {distance_healthy:.4f}) = {weight_healthy:.4f}")
            print(f"{'w_rotten':<10} = exp(-{gamma:<4} * {distance_rotten:.4f}) = {weight_rotten:.4f}")
            print(f"\n3. Final Score Calculation (Healthy Weight Ratio):")
            print(f"{'Formula:':<10} (w_healthy / (w_healthy + w_rotten)) * 100")
            print(f"{'':<10} ({weight_healthy:.4f} / ({weight_healthy:.4f} + {weight_rotten:.4f})) * 100")
            print(f"{'':<10} ({weight_healthy:.4f} / {(weight_healthy + weight_rotten):.4f}) * 100")
            print(f"-"*20)
            print(f"FINAL COLOUR GRADE: {score:.1f}% (Unweighted: {score_ref:1f} )")

    else:
        score = (distance_rotten / (distance_healthy + distance_rotten)) * 100
        summary_lines.append(f"Score: {score:.1f}%   (distance-ratio formulation)")
        if verbose:
            print("\n2. Final Score Calculation (Raw Distance Ratio):")
            print(f"{'Formula:':<10} (d_rotten / (d_healthy + d_rotten)) * 100")
            print(f"{'':<10} ({distance_rotten:.4f} / ({distance_healthy:.4f} + {distance_rotten:.4f})) * 100")
            print(f"{'':<10} ({distance_rotten:.4f} / {(distance_healthy + distance_rotten):.4f}) * 100")
            print(f"FINAL COLOUR GRADE: {score:.1f}%")
    if verbose:
        print("="*50 + "\n")    
    summary_str = "\n".join(summary_lines)
    return round(score, 1), summary_str


def compute_colour_components(fruit_pixels: np.ndarray
                              ) -> tuple[float, float, float]:
    """Compute raw (vibrancy, brightness, uniformity) from HSV fruit pixels.

    Shared by the reference-distribution builder and the grader so the
    same definitions calibrate and score.

    Specular highlights (bright, near-white pixels with V>240 & S<30) are
    filtered out first  - they reflect gloss/water on the produce surface,
    biasing both the saturation mean and std.

    Args:
        fruit_pixels: array of HSV pixel values for the produce region.

    Returns:
        Tuple of raw colour grade components, each in 0 - 1.
    """
    saturation = fruit_pixels[:, 1]
    value = fruit_pixels[:, 2]

    # Drop specular highlights; fall back to raw pixels if nothing remains
    # not_specular = ~((value > 240) & (saturation < 30))
    # if not_specular.any():
    #     saturation = saturation[not_specular]
    #     value = value[not_specular]

    vibrancy = float((saturation.mean() / 255))
    brightness = float(value.mean() / 255)
    uniformity = float(1 - min(saturation.std() / 128, 1.0))
    return vibrancy, brightness, uniformity


def _percentile_score(value: float, sorted_healthy: np.ndarray) -> float:
    """Map a raw component to [0, 1] via the healthy empirical CDF.

    Gets index and determines relative performance against reference.

    A value at the median of the healthy distribution returns 0.5; at
    the max returns ~1.0; below the min returns ~0.0, etc...

    Args:
        value: Raw component value.
        sorted_healthy: Sorted array of healthy-reference component values.

    Returns:
        Percentile rank in (0-1).
    """
    index = np.searchsorted(sorted_healthy, value)
    return float(index / len(sorted_healthy))


def grade_colour_generic(image_path: str | Path, distribution: dict,
                         mask: np.ndarray | None = None) -> tuple:
    """Grade colour for an unknown produce type via distribution func.

    Computes vibrancy, brightness. and saturation uniformity, then maps
    each to a percentile against the distribution of the same component
    across all healthy training images. The reference set defines
    "good".

    Args:
        image_path: Path to the produce image.
        distribution: Dict of sorted healthy arrays under keys
            "vibrancy", "brightness", "uniformity" (from
            build_generic_colour_distribution).
        mask: Optional binary mask (0/1). Generated if not provided.

    Returns:
        Tuple of (Final score, vibrancy_percentile, uniformity_percentile).
    """
    image = cv2.imread(str(image_path))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if mask is None:
        mask = generate_produce_mask(data_path=str(image_path))

    # fruit_pixels = hsv[mask == 1]
    fruit_pixels = hsv[mask > 0]
    if len(fruit_pixels) == 0:
        return 0.0, 0.0, 0.0

    vibrancy, _, _ = compute_colour_components(fruit_pixels)
    vibrancy_pct = _percentile_score(vibrancy, distribution["vibrancy"])

    # Brightness excluded: HSV value is dominated by capture-time exposure
    # rather than intrinsic produce quality.
    colour_score = vibrancy_pct * 100

    return (round(max(0, min(100, colour_score)), 1),
            vibrancy_pct)

def grade_colour(image_path: str | Path, fruit_type: str, references: dict,
                 mask: np.ndarray | None = None) -> float:
    """Score colour against a healthy reference for the given fruit type.

    Uses Bhattacharyya distance to compare the input histogram against the
    reference. HSV captures hue identity; LAB's (a*, b*) plane is
    perceptually uniform and more robust to illumination differences. By
    default the two are fused (mean) to produce a single score.

    Args:
        image_path: Path to the produce image.
        fruit_type: Produce type name, possibly suffixed with health.
        references: Dict of reference histograms from
                    `build_colour_references`. Must contain "median" and,
                    for LAB scoring, "lab_median".
        mask: Optional binary mask (0/1).
        colour_space: "hsv", "lab", or "both" (default). "both" averages
                    the two per-space scores.

    Returns:
        Colour score in [0, 100].
    """
    img = cv2.imread(str(image_path))
    # Scale down excessively large images
    max_dim = 512
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    if mask is None:
        mask = generate_produce_mask(str(image_path))

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    # Ensure mask is the same size as compressed image
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    fruit_key = fruit_type.split("__")[0].strip()
    if fruit_key not in references:
        print(f"'{fruit_key}' not in references {list(references.keys())[:5]}... — returning None")
        return None
    print(f"Looking for: '{fruit_key}'")

    def _score(image: np.ndarray, channels: list[int], bins: list[int],
               ranges: list[int], ref_key: str) -> float:
        hist = cv2.calcHist([image], channels, mask, bins, ranges).astype(np.float32)
        hist /= (hist.sum() + 1e-8)                           # L1 normalise to proper prob. dist.
        ref = references[fruit_key][ref_key].astype(np.float32)
        ref = ref / (ref.sum() + 1e-8)                         # Reference normalised identically
        assert hist.shape == ref.shape, f"hist shape mismatch: {hist.shape} vs {ref.shape}"
        dist = cv2.compareHist(hist, ref, cv2.HISTCMP_BHATTACHARYYA)
        return (1 - dist) * 100

    colour_score = _score(hsv, [0, 1], [30, 32], [0, 180, 0, 256], "mean")

    return round(colour_score, 1)


def visualize_bhattacharyya_score(
    image_path: str | Path,
    fruit_type: str,
    healthy_refs: dict,
    rotten_refs: dict,
    mask: np.ndarray | None = None,
    save_path: str | Path | None = None,
) -> dict:
    """Visualise dual-reference Bhattacharyya scoring for a single image.

    Produces a 2x3 figure: top row shows original/mask/segmented; bottom row
    shows query histogram and the two reference histograms with Bhattacharyya
    distances annotated. The figure title carries the final score.

    Args:
        image_path: Path to the source image.
        fruit_type: Produce type name (key into the references dicts; '__'
            suffix is stripped automatically).
        healthy_refs: Dict mapping fruit-key -> {"median": HSV histogram, ...}
            for healthy training samples.
        rotten_refs: Same shape, for rotten training samples.
        mask: Optional binary mask (0/1). Generated if not provided.
        save_path: Optional path to save the figure (PNG recommended).

    Returns:
        Dict with d_healthy, d_rotten, score, and the matplotlib figure.
    """
    import matplotlib.pyplot as plt

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read {image_path}")

    max_dim = 512
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if mask is None:
        mask = generate_produce_mask(str(image_path))
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    fruit_key = fruit_type.split("__")[0].strip().lower()
    if fruit_key not in healthy_refs:
        raise KeyError(f"'{fruit_key}' missing from healthy_refs (keys: {list(healthy_refs.keys())[:5]}...)")
    if fruit_key not in rotten_refs:
        raise KeyError(f"'{fruit_key}' missing from rotten_refs (keys: {list(rotten_refs.keys())[:5]}...)")

    # Compute query histogram + L1-normalise both query and references identically
    query_hist = cv2.calcHist([hsv], [0, 1], mask, [30, 32], [0, 180, 0, 256]).astype(np.float32)
    query_hist /= (query_hist.sum() + 1e-8)
    h_ref = healthy_refs[fruit_key]["mean"].astype(np.float32)
    h_ref = h_ref / (h_ref.sum() + 1e-8)
    r_ref = rotten_refs[fruit_key]["mean"].astype(np.float32)
    r_ref = r_ref / (r_ref.sum() + 1e-8)

    dist_h = cv2.compareHist(query_hist, h_ref, cv2.HISTCMP_BHATTACHARYYA)
    dist_r = cv2.compareHist(query_hist, r_ref, cv2.HISTCMP_BHATTACHARYYA)
    score = dist_r / (dist_h + dist_r + 1e-8) * 100

    segmented = img_rgb.copy()
    segmented[mask == 0] = 0

    # Shared colour scale across the three histograms for honest comparison
    vmax = max(query_hist.max(), h_ref.max(), r_ref.max())
    extent = [0, 180, 0, 256]   # hue (x) by saturation (y)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle(
        f"Dual-reference unweighted Bhattacharyya scoring - '{fruit_key}'   "
        f"$d_H={dist_h:.3f}$, $d_R={dist_r:.3f}$, "
        f"score $= {score:.1f}\\%$",
        fontsize=12,
    )

    axes[0, 0].imshow(img_rgb);                            axes[0, 0].set_title("Original");  axes[0, 0].axis("off")
    axes[0, 1].imshow(mask, cmap="gray", vmin=0, vmax=1);  axes[0, 1].set_title("Mask");      axes[0, 1].axis("off")
    axes[0, 2].imshow(segmented);                          axes[0, 2].set_title("Segmented"); axes[0, 2].axis("off")

    axes[1, 0].imshow(query_hist.T, origin="lower", aspect="auto", extent=extent, cmap="hot", vmin=0, vmax=vmax)
    axes[1, 0].set_title("Query histogram")
    axes[1, 0].set_xlabel("Hue"); axes[1, 0].set_ylabel("Saturation")

    axes[1, 1].imshow(h_ref.T, origin="lower", aspect="auto", extent=extent, cmap="hot", vmin=0, vmax=vmax)
    axes[1, 1].set_title(f"Healthy reference  ($d={dist_h:.3f}$)")
    axes[1, 1].set_xlabel("Hue"); axes[1, 1].set_ylabel("Saturation")

    axes[1, 2].imshow(r_ref.T, origin="lower", aspect="auto", extent=extent, cmap="hot", vmin=0, vmax=vmax)
    axes[1, 2].set_title(f"Rotten reference  ($d={dist_r:.3f}$)")
    axes[1, 2].set_xlabel("Hue"); axes[1, 2].set_ylabel("Saturation")

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"Saved {save_path}")

    return {
        "fruit_key": fruit_key,
        "dist_healthy": float(dist_h),
        "dist_rotten": float(dist_r),
        "score": float(score),
        "fig": fig,
    }


def grade_proportion(image_path: str | Path, mask: np.ndarray | None = None
                     ) -> float:
    """Score produce shape using contour solidity.

    Solidity (contour area / convex hull area) captures how plump and
    compact the shape is. Low values indicate gaps or imperfections.

    Args:
        image_path: Path to the produce image.
        mask: Optional binary mask (0/1).

    Returns:
        Proportion score in [0, 100].
    """
    if mask is None:
        mask = generate_produce_mask(str(image_path))

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return 0.0

    # Get larger contour (perimeter)
    largest = max(contours, key=cv2.contourArea)

    # Solidity: how does contour fit to convex hull? I.e., plump and full.
    # Convex hull - Perimeter as defined by outermost peaks 
    # Contour     - Actual object perimeter
    convex_hull = cv2.convexHull(largest) 
    contour_area = cv2.contourArea(largest)  
    hull_area = cv2.contourArea(convex_hull)  
    solidity = (contour_area / hull_area) if hull_area > 0 else 0

    # Weighted combination
    proportion_score = solidity * 100
    return round(proportion_score, 1)

if __name__ == "__main__":


    # Load both reference sets
    with open("task_2/data/colour_references_rembg.pkl", "rb") as f:
        healthy_refs = {k.lower(): v for k, v in pickle.load(f).items()}
    with open("task_2/data/colour_references_rembg_rotten.pkl", "rb") as f:
        rotten_refs = {k.lower(): v for k, v in pickle.load(f).items()}

    # Single example
#     result = visualize_bhattacharyya_score(
# "task_2/data/Fruit_And_Vegetable_Diseases_Dataset_no_identical_no_aug/Apple__Healthy/FreshApple (7).jpg",
#         fruit_type="strawberry",
#         healthy_refs=healthy_refs,
#         rotten_refs=rotten_refs,
#         save_path="task_2/figures/dual_score_banana.png",
#     )
#     print(f"d_H={result['dist_healthy']:.3f}, d_R={result['dist_rotten']:.3f}, score={result['score']:.1f}%")



    test_cases = [
        ("task_2/data/Fruit_And_Vegetable_Diseases_Dataset_no_identical_no_aug/Apple__Healthy/FreshApple (7).jpg",        "apple"),   
        ("/home/jaime/AAI/AAI/task_2/data/Fruit_And_Vegetable_Diseases_Dataset_no_identical_no_aug/Apple__Healthy/vertical_flip_Screen Shot 2018-06-08 at 5.18.42 PM.png",  "apple"),   
        ("/home/jaime/AAI/AAI/task_2/data/Fruit_And_Vegetable_Diseases_Dataset_no_identical_no_aug/Apple__Rotten/rottenApple (173).jpg", "apple"),  
    ]
    for path, ftype in test_cases:
        visualize_bhattacharyya_score(
            path, ftype, healthy_refs, rotten_refs,
            save_path=f"task_2/figures/dual_{Path(path).stem}.png",
        )
