"""Render comparative colour-reference plots from pre-built pickles.

Skips the slow build step in `build_colour_references`. Loads existing
reference + distribution pickles and produces:
  1. Per-produce side-by-side mean/median swatch + histogram for each label.
  2. Overall component eCDF comparison across labels.

Usage:
    uv run python utils/compare_references.py
"""
import pickle
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Edit these paths to point at your pre-built pickles ──────────────────────
UTIL_DIR = Path(__file__).resolve().parent
DATA_DIR = UTIL_DIR.parent / "data"
FIG_DIR  = UTIL_DIR.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# {label: (reference_pkl, distribution_pkl)}
SOURCES = {
    "Healthy": (
        DATA_DIR / "colour_references_rembg.pkl",
        DATA_DIR / "generic_colour_distribution.pkl",
    ),
    "Rotten": (
        DATA_DIR / "colour_references_rotten.pkl",
        DATA_DIR / "generic_colour_distribution_rotten.pkl",
    ),
}


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def dominant_rgb(hist):
    """Peak of HSV histogram -> RGB swatch (matches build_references convention)."""
    _, _, _, max_loc = cv2.minMaxLoc(hist.astype(np.float32))
    hue = int(max_loc[1] * 6 + 3)
    sat = int(max_loc[0] * 8 + 4)
    return cv2.cvtColor(np.uint8([[[hue, sat, 220]]]),
                        cv2.COLOR_HSV2RGB)[0][0] / 255.0


# ── Figure 1: per-produce comparison ─────────────────────────────────────────
def plot_per_produce(references_by_label, save_path):
    labels = list(references_by_label.keys())
    all_produce = sorted(set().union(*[r.keys() for r in references_by_label.values()]))
    n_produce = len(all_produce)
    cols_per_label = 3   # mean swatch, median swatch, hist

    fig, axes = plt.subplots(
        n_produce, cols_per_label * len(labels),
        figsize=(2.5 * cols_per_label * len(labels), 2.5 * n_produce),
        squeeze=False,
    )
    fig.suptitle("Colour references — per-produce comparison", fontsize=15)

    for row, produce in enumerate(all_produce):
        for li, label in enumerate(labels):
            refs = references_by_label[label]
            base = li * cols_per_label

            if produce not in refs:
                for c in range(cols_per_label):
                    axes[row, base + c].axis("off")
                continue

            mean_hist   = refs[produce]["mean"]
            median_hist = refs[produce]["median"]
            mean_rgb    = dominant_rgb(mean_hist)
            median_rgb  = dominant_rgb(median_hist)

            # Mean swatch
            axes[row, base].imshow([[mean_rgb]])
            axes[row, base].set_title(f"{label} mean" if row == 0 else "mean",
                                      fontsize=9)
            if li == 0:
                axes[row, base].set_ylabel(produce, fontsize=11, fontweight="bold",
                                            rotation=0, labelpad=45, va="center")
            axes[row, base].set_xticks([]); axes[row, base].set_yticks([])

            # Median swatch
            axes[row, base + 1].imshow([[median_rgb]])
            axes[row, base + 1].set_title(f"{label} median" if row == 0 else "median",
                                           fontsize=9)
            axes[row, base + 1].axis("off")

            # HSV histogram
            axes[row, base + 2].imshow(median_hist.T, origin="lower",
                                        aspect="auto", extent=[0, 180, 0, 256],
                                        cmap="hot")
            axes[row, base + 2].set_title(f"{label} HSV hist" if row == 0 else "HSV hist",
                                           fontsize=9)
            axes[row, base + 2].set_xticks([]); axes[row, base + 2].set_yticks([])

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved {save_path}")
    plt.show()


# ── Figure 2: overall component eCDFs across labels ─────────────────────────
def plot_overall_distributions(distributions_by_label, save_path):
    metrics = ["vibrancy", "brightness", "uniformity"]
    label_colors = {"Healthy": "#10b981", "Rotten": "#d13333"}

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=[m.capitalize() for m in metrics],
                        horizontal_spacing=0.08)

    for col, metric in enumerate(metrics, start=1):
        for label, dist in distributions_by_label.items():
            vals = np.sort(dist.get(metric, []))
            if len(vals) == 0:
                continue
            pct = np.arange(1, len(vals) + 1) / len(vals)
            fig.add_trace(
                go.Scatter(x=vals, y=pct, mode="lines",
                           name=label, legendgroup=label,
                           showlegend=(col == 1),
                           line=dict(color=label_colors.get(label, "black"), width=2.5),
                           hovertemplate=f"{label}<br>{metric}=%{{x:.3f}}<br>pct=%{{y:.2f}}<extra></extra>"),
                row=1, col=col)
            # Median vline
            fig.add_vline(x=float(np.median(vals)),
                          line=dict(color=label_colors.get(label, "black"),
                                    dash="dash", width=1),
                          row=1, col=col, opacity=0.4)
        fig.update_xaxes(title_text=f"Raw {metric.capitalize()}", row=1, col=col)
        fig.update_yaxes(range=[0, 1.05], row=1, col=col)
        if col == 1:
            fig.update_yaxes(title_text="Percentile", row=1, col=col)

    fig.update_layout(
        title=dict(text="<b>Overall colour distribution — Healthy vs Rotten</b>",
                   x=0.5, xanchor="center"),
        height=500, width=1400,
        legend=dict(yanchor="middle", y=0.5, xanchor="left", x=1.02),
        margin=dict(l=60, r=180, t=80, b=60),
    )
    fig.write_html(str(save_path))
    print(f"Saved {save_path}")
    fig.show()


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    references_by_label = {}
    distributions_by_label = {}
    for label, (ref_path, dist_path) in SOURCES.items():
        if not ref_path.exists():
            print(f"[skip] {label}: {ref_path} missing")
            continue
        references_by_label[label] = load(ref_path)
        if dist_path.exists():
            distributions_by_label[label] = load(dist_path)
        print(f"[ok] {label}: {len(references_by_label[label])} produce types")

    if not references_by_label:
        raise FileNotFoundError("No references loaded — check SOURCES paths.")

    plot_per_produce(
        references_by_label,
        save_path=FIG_DIR / "colour_references_compare_per_produce.png",
    )

    if distributions_by_label:
        plot_overall_distributions(
            distributions_by_label,
            save_path=FIG_DIR / "colour_references_compare_overall.html",
        )
