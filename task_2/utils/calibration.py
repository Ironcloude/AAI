"""Softmax calibration measurement: expected calibration error."""

from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from safetensors.torch import load_file
sys.path.append("../..")
from experiment_configs.schema import Experiment

# Number of equal-width confidence bins used by the over-confidence (ECE) test.
# 10 is the standard choice from Guo et al. (2017); fewer bins hide detail,
# more bins yield noisy per-bin accuracy estimates on small val sets.
N_BINS = 10


@torch.no_grad()
def collect_logits(model: torch.nn.Module, dataloader: DataLoader,
                   device: torch.device, is_mtl: bool = False) -> tuple:
    """Forward the model over a dataloader once and return raw logits + labels.

    Args:
        model: Frozen, already-loaded model.
        dataloader: Val-split dataloader yielding (image, health_label, type_label).
        device: Device the model sits on.
        is_mtl: True if the model returns (health_logits, type_logits).

    Returns:
        If MTL: (health_logits, type_logits, health_labels, type_labels).
        Else:   (health_logits, health_labels).
    """
    model.eval()
    health_logits, type_logits = [], []
    health_labels, type_labels = [], []
    for x_batch, y_health, y_type in dataloader:
        x_batch = x_batch.to(device)
        out = model(x_batch)
        if is_mtl:
            h_out, t_out = out
            type_logits.append(t_out.cpu())
            type_labels.append(y_type)
        else:
            h_out = out
        health_logits.append(h_out.cpu())
        health_labels.append(y_health)
    health_logits = torch.cat(health_logits)
    health_labels = torch.cat(health_labels)
    if is_mtl:
        return (health_logits, torch.cat(type_logits),
                health_labels, torch.cat(type_labels))
    return health_logits, health_labels


# Calculate Expected Calibration Error for prediction confidence.
def compute_ece(logits: torch.Tensor, labels: torch.Tensor,
                n_bins: int = N_BINS,
                min_confidence: float | None = None) -> tuple[float, dict]:
    """Expected Calibration Error via equal-width confidence bins.

    For each bin, compare the average "predicted confidence max" to the
    "empirical accuracy" of samples in that bin. A well-calibrated model
    has them equal; ECE is the sample-weighted mean absolute gap.


    Args:
        logits: Raw model logits, shape (N, C).
        labels: Ground-truth class indices, shape (N,).
        n_bins: Number of equal-width confidence bins.
        min_confidence: Lower edge of the first bin. Defaults to 1 / C, the
            theoretical minimum of softmax's argmax confidence — this keeps
            bins full for binary (where max(p) >= 0.5) and avoids wasting
            resolution on unreachable ranges.

    Returns:
        (ece, stats) where stats has midpoints / confidences / accuracies /
        counts arrays plus the min/max confidence range, for plotting.
    """
    probs = torch.softmax(logits, dim=1)
    # confidences = [0.90, 0.70, 0.55]   # peak value per row
    # predictions = [  0,    1,    0 ]   # argmax per row
    confidences, predicted_classes = probs.max(dim=1) 
    accuracies = (predicted_classes == labels).float()

    if min_confidence is None:
        min_confidence = 1.0 / logits.shape[1]

    bin_boundaries = torch.linspace(min_confidence, 1.0, n_bins + 1)
    ece_gap = 0.0
    bin_confs, bin_accs, bin_counts = [], [], []
    for low, high in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (confidences > low) & (confidences <= high)
        count = in_bin.float().sum().item()
        if count > 0:
            # Mean confidence of samples in this bin
            # 'What is the mean of the predicted confidences in this bin?'
            bin_conf = confidences[in_bin].mean().item()
            # Mean actual accuracy of samples in this bin
            # 'What % of predictions are actually correct in this bin?'
            bin_acc = accuracies[in_bin].mean().item()
            # Weight this bin's gap by its relative frequency in the dataset
            ece_gap += (count / len(labels)) * abs(bin_acc - bin_conf)
        else:
            bin_conf = ((low + high) / 2).item()
            bin_acc = 0.0
        bin_confs.append(bin_conf)
        bin_accs.append(bin_acc)
        bin_counts.append(count)

    midpoints = 0.5 * (bin_boundaries[:-1] + bin_boundaries[1:]).numpy()
    return ece_gap, {
        "midpoints": midpoints,
        "confidences": np.array(bin_confs),
        "accuracies": np.array(bin_accs),
        "counts": np.array(bin_counts),
        "min_confidence": float(min_confidence),
        "max_confidence": 1.0,
    }


def add_reliability_traces(fig, stats, row, col, show_legend):
    """Draw perfect-calibration diagonal plus observed-accuracy and miscalibration-gap bars."""
    midpoints = stats["midpoints"]
    confidences = stats["confidences"]
    accuracies = stats["accuracies"]
    counts = stats["counts"]
    lo = stats["min_confidence"]
    hi = stats["max_confidence"]
    nonempty = counts > 0
    bar_width = (hi - lo) / len(counts)

    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line=dict(color="black", dash="dash"),
        name="Perfect calibration", legendgroup="perfect",
        showlegend=show_legend,
    ), row=row, col=col)

    fig.add_trace(go.Bar(
        x=midpoints[nonempty], y=accuracies[nonempty], width=bar_width,
        marker=dict(color="steelblue", line=dict(color="black", width=1)),
        opacity=1, name="Observed accuracy", legendgroup="observed",
        showlegend=show_legend,
    ), row=row, col=col)

    gap = confidences[nonempty] - accuracies[nonempty]
    fig.add_trace(go.Bar(
        x=midpoints[nonempty], y=gap, base=accuracies[nonempty], width=bar_width,
        marker=dict(color="red", line=dict(color="red", width=1)),
        opacity=0.35, name="Miscalibration gap", legendgroup="gap",
        showlegend=show_legend,
    ), row=row, col=col)

    fig.update_xaxes(title_text="Predicted confidence", range=[lo, hi], row=row, col=col)
    fig.update_yaxes(title_text="Empirical accuracy", range=[0, 1], row=row, col=col)


def create_calibration_plots(model: torch.nn.Module, model_save_name: str | None,
                            test_model: str | None,  
                            val_loader: torch.utils.data.DataLoader, 
                            device: torch.device , experiment: Experiment):
    """Create Expected Calibration Error (ECE) calibration plots 
    for the given model and validation dataloader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        model_save_name (str | None): Optional. The name of the running model to load.
        test_model (str | None): Optional. The name of the specified model to test
        val_loader (torch.utils.data.DataLoader): The validation loader.
        device (torch.device): The device to run the model on.
        experiment (Experiment): The corresponding experiment configuration.
    """
    model.load_state_dict(load_file(
        f"models/{model_save_name if not test_model else test_model}.safetensors"))

    if experiment.is_mtl:
        heal_logits, type_logits, health_labels, type_labels = collect_logits(
            model, val_loader, device, is_mtl=True)
        health_ece, health_stats = compute_ece(heal_logits, health_labels, n_bins=N_BINS)
        type_ece, type_stats = compute_ece(type_logits, type_labels, n_bins=N_BINS)
        print(f"Health head ECE: {health_ece:.4f}")
        print(f"Type head   ECE: {type_ece:.4f}")

        fig = make_subplots(
            rows=1, cols=2, horizontal_spacing=0.12,
            subplot_titles=(f"Health head (ECE = {health_ece:.4f})",
                            f"Type head (ECE = {type_ece:.4f})"),
        )
        add_reliability_traces(fig, health_stats, row=1, col=1, show_legend=True)
        add_reliability_traces(fig, type_stats, row=1, col=2, show_legend=False)
    else:
        heal_logits, health_labels = collect_logits(model, val_loader, device, is_mtl=False)
        health_ece, health_stats = compute_ece(heal_logits, health_labels, n_bins=N_BINS)
        print(f"Health head ECE: {health_ece:.4f}")

        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=(f"Health head (ECE = {health_ece:.4f})",),
        )
        add_reliability_traces(fig, health_stats, row=1, col=1, show_legend=True)

    fig.update_layout(
        title=dict(text=f"{experiment.display_name} — calibration on val split",
                x=0.5, xanchor="center"),
        barmode="overlay", width=1100 if experiment.is_mtl else 600, height=550,
    )
    fig.show()

    figures_dir = Path(__file__).resolve().parent.parent / "figures"
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / f"calibration_ece{experiment.display_name}.png"
    fig.write_image(fig_path, scale=2)  