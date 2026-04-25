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


# Calculate colour features used for produce quality scoring.
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

    vibrancy, _, uniformity = compute_colour_components(fruit_pixels)
    print(f"RAW - Vib: {vibrancy:.2f}, Uni: {uniformity:.2f}")
    vibrancy_pct = _percentile_score(vibrancy, distribution["vibrancy"])
    uniformity_pct = _percentile_score(uniformity, distribution["uniformity"])

    # Brightness excluded: HSV value is dominated by capture-time exposure
    # rather than intrinsic produce quality.
    colour_score = (vibrancy_pct + uniformity_pct) / 2 * 100

    return (round(max(0, min(100, colour_score)), 1),
            vibrancy_pct, uniformity_pct)

def grade_colour(image_path: str | Path, fruit_type: str, references: dict,
                 mask: np.ndarray | None = None) -> float:
    """Score colour against a healthy reference for the given fruit type.

    Uses Bhattacharyya distance to compare
    the input's HSV histogram against the reference.

    Args:
        image_path: Path to the produce image.
        fruit_type: Produce type name, possibly suffixed with health.
        references: Dict of reference histograms from
                    `build_colour_references`.
        mask: Optional binary mask (0/1).
        verbose: If True, print diagnostic details.

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
    if mask is None:
        mask = generate_produce_mask(str(image_path))

    # Ensure mask is the same size as compressed image
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Calculate frequency distribution of pixel values
    # More bins; more precision.
    histogram = cv2.calcHist(
        [hsv],              # image
        [0, 1],             # Channels for hue and saturation
        mask,
        [30, 32],           # bin counts (hue, saturation)
        [0, 180, 0, 256]    # 0-180 (hue); 0-256 (saturation)
    )
    # Scale histogram to 1 (larger images don't dominate; proportional)
    cv2.normalize(histogram, histogram)

    # Compute raw hist comparison to average
    fruit_key = fruit_type.split("__")[0].strip() # replace __rotten with __healthy
    print(f"Looking for: '{fruit_key}'")
    print(f"Available: {list(references.keys())}")
    reference_histogram = references[fruit_key]["median"]
    dist = cv2.compareHist(
        histogram, reference_histogram, 
        cv2.HISTCMP_BHATTACHARYYA #More forgiving, measures overlap
        # cv2.HISTCMP_CORREL # Strict, penalises minor variations
    )
    # CORREL returns -1 to 1, map to 0-100
    # color_score = max(0, similarity) * 100
    color_score = ((1 - dist))  * 100

    return round(color_score, 1) 


# Score produce shape using mask contour solidity.
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
