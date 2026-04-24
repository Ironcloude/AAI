"""Colour and proportion grading functions for produce quality assessment."""

import sys
from pathlib import Path
import cv2
import numpy as np
sys.path.append(".")
try:
    from utils.generate_masks import generate_produce_mask
except ModuleNotFoundError:
    from generate_masks import generate_produce_mask

    
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

    h_ref = _norm(healthy_refs[fruit_key]["median"])
    r_ref = _norm(rotten_refs[fruit_key]["median"])
    distance_healthy = cv2.compareHist(hist, h_ref, cv2.HISTCMP_BHATTACHARYYA)
    distance_rotten = cv2.compareHist(hist, r_ref, cv2.HISTCMP_BHATTACHARYYA)
    if distance_healthy + distance_rotten == 0: # Division by zero gaurd
        return None
    
    # for plot display
    summary_lines = [f"d_H:{distance_healthy:.3f} | d_R:{distance_rotten:.3f}"]
    
    if verbose:
        print("\n" + "="*50)
        print(f"DUAL COLOUR GRADING: {fruit_key.upper()}")
        print("-" * 50)
        print("1. Raw Bhattacharyya Distances (Lower is better):")
        print(f"{'d_healthy':<10} = {distance_healthy:.4f}")
        print(f"{'d_rotten':<10} = {distance_rotten:.4f}")

    if gamma:
        weight_healthy = np.exp(-gamma * distance_healthy)
        weight_rotten = np.exp(-gamma * distance_rotten)
        
        score = (weight_healthy / (weight_healthy + weight_rotten)) * 100
        score_ref = (distance_rotten / (distance_healthy + distance_rotten)) * 100

        summary_lines.append(f"w_H:{weight_healthy:.3f} | w_R:{weight_rotten:.3f} (g={gamma})")
        summary_lines.append(f"Score:{score:.1f}% (Raw:{score_ref:.1f}%)")
        
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
        summary_lines.append(f"Score:{score:.1f}%")
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
    not_specular = ~((value > 240) & (saturation < 30))
    if not_specular.any():
        saturation = saturation[not_specular]
        value = value[not_specular]

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

    colour_score = _score(hsv, [0, 1], [30, 32], [0, 180, 0, 256], "median")

    return round(colour_score, 1)


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
