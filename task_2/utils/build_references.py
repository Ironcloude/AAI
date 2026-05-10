"""Colour reference building and visualisation for produce grading."""
from pathlib import Path
import pickle
import sys
import cv2
from matplotlib import pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
sys.path.append(".")
from generate_masks import generate_produce_mask
from grade_produce import compute_colour_components



def build_colour_references(dataset_root: str | Path, mask_root: str | Path = None,
                            example: bool = False, 
                            save_paths: list[str] = ["colour_reference.pkl",
                                                     "colour_reference.pkl",
                                                     "generic_colour_distribution.pkl"],
                            rotten: bool = False) -> dict:
    """Build per-fruit-type reference HSV and LAB histograms and colour
    disitributions from healthy training images.

    The histograms serve as a baseline reference for colour scoring of known fruit
    at inference time. HSV (hue, saturation) captures colour-identity;
    LAB (a*, b*) captures perceptual chroma and is more robust to illumination
    shifts, which helps when comparing produce across lighting conditions.

    The colour distribution reference serves in normalising similarity scores;
    raw metrics skew towards a lower baseline.

    Args:
        dataset_root: Path to produce dataset directory.
        mask_root: Path to produce mask directory.
        example: Single directory mode for testing.
        save_path: Output path for reference and distrubtion pickle files.

    Returns:
        Dictionary of reference histograms (HSV + LAB) and hsv distribution.
    """

    references = {}
    dataset_root = Path(dataset_root)
    mask_root = Path(mask_root) if mask_root else None

    if example:
        reference_directory = [dataset_root]
    elif not rotten:
        reference_directory = [
            directory for directory in dataset_root.iterdir()
            if directory.is_dir() and "healthy" in directory.name.lower()
        ]
    else: 
        reference_directory = [
        directory for directory in dataset_root.iterdir()
        if directory.is_dir() and "rotten" in directory.name.lower()
        ]
    produce_classes = {}
    # For each healthy produce image, find its corresponding mask
    for produce_directory in reference_directory:
        cls_vibrancy, cls_brightness, cls_uniformity = [], [], []
        fruit_type = produce_directory.name.split("__")[0].strip().lower()
        histograms = []
        lab_histograms = []
        drop_count = 0
        # For each produce
        for image_path in produce_directory.iterdir():
            if image_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                continue
            
            # Read into np array BGR order
            # Convert BGR (Blue:Green:Red) to HSV (Hue:Saturation:Value)
            # and LAB (Lightness:a*[green-red]:b*[blue-yellow])
            produce_image = cv2.imread(str(image_path))
            if produce_image is None:
                continue
            hsv_produce_image = cv2.cvtColor(produce_image, cv2.COLOR_BGR2HSV)
            lab_produce_image = cv2.cvtColor(produce_image, cv2.COLOR_BGR2LAB)

            # Load pre-computed mask if available, otherwise fallback to rembg
            mask = None
            if mask_root:
                mask_path = mask_root / produce_directory.name / \
                            (image_path.stem + ".png")
                if mask_path.exists():
                    # Loads as (H, W), values 0 - 255 
                    # (background [0] or foreground [255])
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    # Convert to 0/1 for calcHist. 
                    # Anything >127 is foreground; <=127 is background
                    # mask                          [0, 255, 0, 128]
                    # mask > 127                    [False, True, False, True]
                    # (mask > 127).astype(np.uint8) [0, 1, 0,  1]
                    mask = (mask > 127).astype(np.uint8)
                    # Ensure mask matches image dimensions
                    h, w = produce_image.shape[:2]
                    if mask.shape != (h, w):
                        mask = cv2.resize(mask, (w, h), 
                                          interpolation=cv2.INTER_NEAREST)
            if mask is None:
                mask = generate_produce_mask(data_path=str(image_path), 
                                             method="rembg")

            # Calculate frequency distribution of pixel values
            # More bins; more precision.
            hist = cv2.calcHist(
                images=[hsv_produce_image],
                channels=[0, 1],        # Channels for hue and saturation
                mask=mask,
                histSize=[30, 32],      # bin counts (hue, saturation).
                ranges=[0, 180, 0, 256] # 0-180 (hue); 0-256 (saturation)
            )
            # Scale histogram to 1 (larger images don't dominate; proportional)
            cv2.normalize(hist, hist)

            # Skip if dominant colour (max loc) is very low saturation
            # Likely faulty mask isolating background
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(hist)
            # remove saturation bin < 5 out of 32 == 16% of lower end
            if not rotten and max_loc[0] < 5:
                drop_count += 1
                continue

            fruit_pixels = hsv_produce_image[mask == 1]
            vibrancy, brightness, uniformity = compute_colour_components(fruit_pixels)
            cls_vibrancy.append(vibrancy)
            cls_brightness.append(brightness)
            cls_uniformity.append(uniformity)

            histograms.append(hist)

        print(f"{produce_directory.name}: {len(cls_vibrancy)} samples, {drop_count} dropped")
        
        if histograms:
            stacked = np.array(histograms) # stack all histograms for mean/media ops
            references[fruit_type] = {
                "mean": np.mean(stacked, axis=0).astype(np.float32),
                "median": np.median(stacked, axis=0).astype(np.float32),
            }

        produce_classes[fruit_type] = {"vibrancy": cls_vibrancy, "brightness": cls_brightness, "uniformity": cls_uniformity}

    # Ensure class weighting for distribution (otherwise banana's would dominate, etc.)
    rng = np.random.default_rng(seed=42)
    min_count = min(len(cls["vibrancy"]) for cls in produce_classes.values())
    vibrancies, brightnesses, uniformities = [], [], []
    for components in produce_classes.values():
        index = rng.choice(len(components["vibrancy"]), size=min_count, replace=False)
        vibrancies.extend(np.array(components["vibrancy"])[index])
        brightnesses.extend(np.array(components["brightness"])[index])
        uniformities.extend(np.array(components["uniformity"])[index])
    print(f"Balanced to {min_count} samples per class * {len(produce_classes)} types")

    distribution = {
        "vibrancy": np.sort(np.array(vibrancies, dtype=np.float32)),
        "brightness": np.sort(np.array(brightnesses, dtype=np.float32)),
        "uniformity": np.sort(np.array(uniformities, dtype=np.float32)),
        # Per-type unsorted arrays for plotting
        "per_type": {
            fruit_type: {
                metric: np.array(components[metric], dtype=np.float32)
                for metric in ("vibrancy", "brightness", "uniformity")
            }
            for fruit_type, components in produce_classes.items()
        },
    }

    if not example:
        for i, save_path in enumerate(save_paths):
            output_path = Path(save_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                pickle.dump(references if i == 0 else distribution, f)
                
            print(f"References successfully saved to {output_path}")
            print(f"{list(references.keys()) if i == 0 else f'({len(vibrancies)} samples)'}")

    return references, distribution


def plot_colour_references(ref_path: str | Path, dist_path: str | Path,
                           figures_dir: str | Path | None = None) -> None:
    """Visualise saved colour references and score distributions.

    Produces two separate figures:
      1. Per-fruit mean colour, median colour, and median HSV histogram.
      2. Per-type eCDFs overlaid on the balanced global eCDF for each
         component (vibrancy, brightness, uniformity).

    Args:
        ref_path: Path to a hist reference pickle file.
        dist_path: Path to a distribution pickle file.
        figures_dir: Optional output directory for saving PNGs. Defaults
            to ``<util_parent>/figures``.
    """
    with open(ref_path, "rb") as f:
        references = pickle.load(f)
    with open(dist_path, "rb") as f:
        distributions = pickle.load(f)

    if figures_dir is None:
        figures_dir = Path(__file__).resolve().parent.parent / "figures"
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    def dominant_rgb(hist):
        _, _, _, max_loc = cv2.minMaxLoc(hist.astype(np.float32))
        hue = int(max_loc[1] * 6 + 3)
        sat = int(max_loc[0] * 8 + 4)
        return cv2.cvtColor(np.uint8([[[hue, sat, 220]]]),
                            cv2.COLOR_HSV2RGB)[0][0] / 255.0, hue, sat

    def dominant_lab_rgb(hist, L=200):
        # hist is indexed [a_bin, b_bin]; minMaxLoc returns (col, row) = (b, a)
        _, _, _, max_loc = cv2.minMaxLoc(hist.astype(np.float32))
        a = int(max_loc[1] * 8 + 4)
        b = int(max_loc[0] * 8 + 4)
        rgb = cv2.cvtColor(np.uint8([[[L, a, b]]]),
                           cv2.COLOR_LAB2RGB)[0][0] / 255.0
        return rgb, a, b

    # Figure 1: per-fruit colour references (HSV + LAB side-by-side)
    n = len(references)
    # 5 cols: mean HSV, median HSV, HSV hist, median LAB, LAB hist.
    fig1, axes1 = plt.subplots(n, 5, figsize=(20, 3.2 * n), squeeze=False)
    fig1.suptitle("Healthy Colour References (per fruit type)", fontsize=16)

    for row, (fruit_type, data) in enumerate(sorted(references.items())):
        mean_hist = data["mean"]
        median_hist = data["median"]
        lab_median_hist = data["lab_median"]

        mean_rgb, mean_h, mean_s = dominant_rgb(mean_hist)
        median_rgb, med_h, med_s = dominant_rgb(median_hist)
        lab_rgb, lab_a, lab_b = dominant_lab_rgb(lab_median_hist)

        # Fruit type sits as a row label on the left, keeping column titles short
        axes1[row, 0].imshow([[mean_rgb]])
        axes1[row, 0].set_title(f"HSV Mean\nH={mean_h}  S={mean_s}", fontsize=10)
        axes1[row, 0].set_ylabel(fruit_type, fontsize=11, fontweight="bold",
                                 rotation=0, labelpad=45, va="center")
        axes1[row, 0].set_xticks([])
        axes1[row, 0].set_yticks([])

        axes1[row, 1].imshow([[median_rgb]])
        axes1[row, 1].set_title(f"HSV Median\nH={med_h}  S={med_s}", fontsize=10)
        axes1[row, 1].axis("off")

        axes1[row, 2].imshow(median_hist.T, origin="lower", aspect="auto",
                             extent=[0, 180, 0, 256], cmap="hot")
        axes1[row, 2].set_title("Median HSV Histogram", fontsize=10)
        axes1[row, 2].set_ylabel("Sat")
        if row == n - 1:
            axes1[row, 2].set_xlabel("Hue")

        axes1[row, 3].imshow([[lab_rgb]])
        axes1[row, 3].set_title(f"LAB Median\na={lab_a}  b={lab_b}", fontsize=10)
        axes1[row, 3].axis("off")

        axes1[row, 4].imshow(lab_median_hist.T, origin="lower", aspect="auto",
                             extent=[0, 256, 0, 256], cmap="hot")
        axes1[row, 4].set_title("Median LAB Histogram", fontsize=10)
        axes1[row, 4].set_ylabel("b*")
        if row == n - 1:
            axes1[row, 4].set_xlabel("a*")

    fig1.tight_layout(rect=[0, 0, 1, 0.97])
    fig1_path = figures_dir / "colour_references_per_type.png"
    fig1.savefig(fig1_path, dpi=150, bbox_inches="tight")
    print(f"Saved {fig1_path}")

    # Figure 2: per-type vs balanced global eCDFs (plotly, interactive)
    dist_metrics = ["vibrancy", "brightness", "uniformity"]
    per_type = distributions.get("per_type", {})
    # Hex colours from matplotlib's tab20 for consistency with Figure 1
    cmap = plt.cm.tab20(np.linspace(0, 1, max(len(per_type), 1)))
    type_colors = ["#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
                   for r, g, b, _ in cmap]

    fig2 = make_subplots(
        rows=1, cols=3,
        subplot_titles=[m.capitalize() for m in dist_metrics],
        horizontal_spacing=0.08,
    )

    for col, metric in enumerate(dist_metrics, start=1):
        global_scores = distributions.get(metric, [])
        if len(global_scores) == 0:
            continue

        # Per-type eCDFs — grouped in legend so one click toggles all panels for a fruit
        for (fruit_type, comps), color in zip(sorted(per_type.items()), type_colors):
            type_scores = np.sort(comps[metric])
            type_percentiles = np.arange(1, len(type_scores) + 1) / len(type_scores)
            fig2.add_trace(
                go.Scatter(
                    x=type_scores, y=type_percentiles,
                    mode="lines",
                    line=dict(color=color, width=1),
                    opacity=0.6,
                    name=fruit_type,
                    legendgroup=fruit_type,
                    showlegend=(col == 1),
                    hovertemplate=f"{fruit_type}<br>{metric}=%{{x:.3f}}<br>pct=%{{y:.2f}}<extra></extra>",
                ),
                row=1, col=col,
            )

        # Balanced global eCDF
        sorted_global = np.sort(global_scores)
        global_percentiles = np.arange(1, len(sorted_global) + 1) / len(sorted_global)
        fig2.add_trace(
            go.Scatter(
                x=sorted_global, y=global_percentiles,
                mode="lines",
                line=dict(color="black", width=2.5),
                name="Balanced global",
                legendgroup="global",
                showlegend=(col == 1),
                hovertemplate=f"balanced<br>{metric}=%{{x:.3f}}<br>pct=%{{y:.2f}}<extra></extra>",
            ),
            row=1, col=col,
        )

        # Median vertical line
        median_val = float(np.median(sorted_global))
        fig2.add_vline(
            x=median_val, line=dict(color="red", dash="dash", width=1),
            annotation_text=f"median {median_val:.2f}",
            annotation_position="top right",
            row=1, col=col,
        )

        fig2.update_xaxes(title_text=f"Raw {metric.capitalize()}", row=1, col=col)
        fig2.update_yaxes(range=[0, 1.05], row=1, col=col)
        if col == 1:
            fig2.update_yaxes(title_text="Percentile", row=1, col=col)

    fig2.update_layout(
        title="Generic Colour Distributions: per-type vs balanced global eCDF",
        height=500, width=1400,
        legend=dict(yanchor="middle", y=0.5, xanchor="left", x=1.02, font=dict(size=10)),
        margin=dict(l=60, r=180, t=80, b=60),
    )

    fig2_html_path = figures_dir / "generic_colour_ecdfs.html"
    fig2.write_html(str(fig2_html_path))
    print(f"Saved {fig2_html_path}")

    # Figures are already saved to disk above; skip interactive display when
    # running headless (otherwise plt.show() blocks the pipeline).
    # plt.show()
    # fig2.show()


def plot_healthy_vs_rotten_references(healthy_ref_path: str | Path,
                                      rotten_ref_path: str | Path,
                                      figures_dir: str | Path | None = None) -> None:
    """Compare per-type healthy vs rotten HSV references side by side.

    For each produce type present in both reference pickles, renders a row
    with mean and median swatches and HSV histograms for healthy and rotten
    (eight columns total). Types missing from either side are skipped.

    Args:
        healthy_ref_path: Path to healthy references pickle.
        rotten_ref_path:  Path to rotten references pickle.
        figures_dir: Optional output directory. Defaults to ../figures.
    """
    with open(healthy_ref_path, "rb") as f:
        healthy_refs = pickle.load(f)
    with open(rotten_ref_path, "rb") as f:
        rotten_refs = pickle.load(f)

    if figures_dir is None:
        figures_dir = Path(__file__).resolve().parent.parent / "figures"
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Shared types only — need both sides to compare
    shared = sorted(set(healthy_refs.keys()) & set(rotten_refs.keys()))
    if not shared:
        print("No shared fruit types between healthy and rotten references")
        return

    def dominant_rgb(hist):
        _, _, _, max_loc = cv2.minMaxLoc(hist.astype(np.float32))
        hue = int(max_loc[1] * 6 + 3)
        sat = int(max_loc[0] * 8 + 4)
        return cv2.cvtColor(np.uint8([[[hue, sat, 220]]]),
                            cv2.COLOR_HSV2RGB)[0][0] / 255.0, hue, sat

    n = len(shared)
    # 8 columns per row:
    # 0: H mean swatch | 1: H median swatch | 2: H mean hist | 3: H median hist
    # 4: R mean swatch | 5: R median swatch | 6: R mean hist | 7: R median hist
    fig, axes = plt.subplots(n, 8, figsize=(22, 2.6 * n), squeeze=False)
    fig.suptitle("Healthy vs Rotten Colour References — mean and median (per fruit type)",
                 fontsize=15)

    for row, fruit_type in enumerate(shared):
        h_mean = healthy_refs[fruit_type]["mean"]
        h_med  = healthy_refs[fruit_type]["median"]
        r_mean = rotten_refs[fruit_type]["mean"]
        r_med  = rotten_refs[fruit_type]["median"]

        h_mean_rgb, h_mean_h, h_mean_s = dominant_rgb(h_mean)
        h_med_rgb,  h_med_h,  h_med_s  = dominant_rgb(h_med)
        r_mean_rgb, r_mean_h, r_mean_s = dominant_rgb(r_mean)
        r_med_rgb,  r_med_h,  r_med_s  = dominant_rgb(r_med)

        # 0: Healthy mean swatch (carries the row label on the left)
        axes[row, 0].imshow([[h_mean_rgb]])
        axes[row, 0].set_title(f"H mean\nH={h_mean_h}  S={h_mean_s}", fontsize=9)
        axes[row, 0].set_ylabel(fruit_type, fontsize=11, fontweight="bold",
                                 rotation=0, labelpad=45, va="center")
        axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])

        # 1: Healthy median swatch
        axes[row, 1].imshow([[h_med_rgb]])
        axes[row, 1].set_title(f"H median\nH={h_med_h}  S={h_med_s}", fontsize=9)
        axes[row, 1].axis("off")

        # 2: Healthy mean histogram
        axes[row, 2].imshow(h_mean.T, origin="lower", aspect="auto",
                            extent=[0, 180, 0, 256], cmap="hot")
        axes[row, 2].set_title("H mean hist", fontsize=9)
        axes[row, 2].set_ylabel("Sat")
        if row == n - 1:
            axes[row, 2].set_xlabel("Hue")

        # 3: Healthy median histogram
        axes[row, 3].imshow(h_med.T, origin="lower", aspect="auto",
                            extent=[0, 180, 0, 256], cmap="hot")
        axes[row, 3].set_title("H median hist", fontsize=9)
        axes[row, 3].set_yticks([])
        if row == n - 1:
            axes[row, 3].set_xlabel("Hue")

        # 4: Rotten mean swatch
        axes[row, 4].imshow([[r_mean_rgb]])
        axes[row, 4].set_title(f"R mean\nH={r_mean_h}  S={r_mean_s}", fontsize=9)
        axes[row, 4].axis("off")

        # 5: Rotten median swatch
        axes[row, 5].imshow([[r_med_rgb]])
        axes[row, 5].set_title(f"R median\nH={r_med_h}  S={r_med_s}", fontsize=9)
        axes[row, 5].axis("off")

        # 6: Rotten mean histogram
        axes[row, 6].imshow(r_mean.T, origin="lower", aspect="auto",
                            extent=[0, 180, 0, 256], cmap="hot")
        axes[row, 6].set_title("R mean hist", fontsize=9)
        axes[row, 6].set_ylabel("Sat")
        if row == n - 1:
            axes[row, 6].set_xlabel("Hue")

        # 7: Rotten median histogram
        axes[row, 7].imshow(r_med.T, origin="lower", aspect="auto",
                            extent=[0, 180, 0, 256], cmap="hot")
        axes[row, 7].set_title("R median hist", fontsize=9)
        axes[row, 7].set_yticks([])
        if row == n - 1:
            axes[row, 7].set_xlabel("Hue")

    # Report any types missing from one side — interesting to note in writeup
    only_healthy = sorted(set(healthy_refs.keys()) - set(rotten_refs.keys()))
    only_rotten  = sorted(set(rotten_refs.keys()) - set(healthy_refs.keys()))
    if only_healthy:
        print(f"Types only in healthy (skipped): {only_healthy}")
    if only_rotten:
        print(f"Types only in rotten (skipped):  {only_rotten}")

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_path = figures_dir / "healthy_vs_rotten_references.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rotten", action="store_true")
    parser.add_argument("--compare", action="store_true",
                        help="Skip build; just plot healthy vs rotten comparison")
    args = parser.parse_args()

    util_dir = Path(__file__).resolve().parent
    dataset_dir = util_dir.parent / "data" / "Fruit_And_Vegetable_Diseases_Dataset"
    mask_dir = util_dir.parent / "data" / "rembg_masks"
    healthy_ref_path = util_dir.parent / "data" / "colour_references_rembg.pkl"
    rotten_ref_path  = util_dir.parent / "data" / "colour_references_rembg_rotten.pkl"
    dist_path = util_dir.parent / "data" / "generic_colour_distribution.pkl"

    if args.compare:
        plot_healthy_vs_rotten_references(str(healthy_ref_path), str(rotten_ref_path))
    elif args.rotten:
        print("Building rotten colour references...")
        build_colour_references(dataset_dir, mask_root=mask_dir,
                                save_paths=[str(rotten_ref_path), "/tmp/dist_rotten_unused.pkl"],
                                rotten=True)
    else:
        print("Building healthy colour references...")
        build_colour_references(dataset_dir, mask_root=mask_dir,
                                save_paths=[str(healthy_ref_path), str(dist_path)], rotten=False)
        plot_colour_references(str(healthy_ref_path), str(dist_path))

    # import cv2
    # from pathlib import Path
    # import random

    # rotten_dir = Path(".") / "data" / "Fruit_And_Vegetable_Diseases_Dataset_no_identical_no_aug" / "Cucumber__Rotten"
    # samples = random.sample(list(rotten_dir.iterdir()), 10)

    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    # for ax, p in zip(axes.flat, samples):
    #     img = cv2.imread(str(p))
    #     ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #     ax.set_title(p.name[:20], fontsize=7)
    #     ax.axis("off")
    # plt.tight_layout()
    # plt.show()
