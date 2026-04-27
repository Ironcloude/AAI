import os
import json
from dataclasses import asdict
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def save_run_data(run_config, metrics, output_dir="results"):
    """Saves Config and Metrics combined into a clean JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = Path(output_dir) / f"{run_config.run}.json"

    combined = {
        "config": asdict(run_config),
        "metrics": asdict(metrics)
    }

    with open(filepath, "w") as f:
        json.dump(combined, f, indent=4)
    print(f"Data saved to {filepath}")


def get_markdown_text(run_config, metrics):
    """Generates the Markdown string containing tables for hyperparameters and metrics."""
    conf_dict = asdict(run_config)
    met_dict = asdict(metrics)

    run_name = conf_dict.pop('run', 'Unknown')
    met_dict.pop('run', None)

    lines = [f"# Run Overview: {run_name}", ""]

    # --- Tunable Hyperparameters Table ---
    lines.extend([
        "##### Tunable hyperparameters", "",
        f"| {' | '.join(conf_dict.keys())} |",
        f"| {' | '.join(['---'] * len(conf_dict))} |",
        f"| {' | '.join([str(v) for v in conf_dict.values()])} |",
        ""
    ])

    # --- Metrics Table ---
    # Filter out lists for the summary table
    summary_metrics = {k: v for k,
                       v in met_dict.items() if not isinstance(v, list)}

    if summary_metrics:
        lines.extend([
            "##### Evaluation Metrics", "",
            f"| {' | '.join(summary_metrics.keys())} |",
            f"| {' | '.join(['---'] * len(summary_metrics))} |",
            f"| {' | '.join([str(round(v, 4)) if isinstance(v, float) else str(v) for v in summary_metrics.values()])} |",
            ""
        ])

    return "\n".join(lines)


def build_html_template(plot_div, markdown_text, title):
    """Wraps Plotly HTML and Markdown text into a single cohesive HTML page with a copy button."""
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f5f6fa; }}
        .dashboard-container {{ max-width: 1200px; margin: auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }}
        .md-container {{ background: #282a36; color: #f8f8f2; padding: 25px; border-radius: 8px; position: relative; margin-top: 30px; box-shadow: inset 0 2px 4px rgba(0,0,0,0.5); }}
        .copy-btn {{ position: absolute; top: 15px; right: 15px; padding: 8px 16px; background: #50fa7b; color: #282a36; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; font-size: 14px; transition: background 0.3s; }}
        .copy-btn:hover {{ background: #40c963; }}
        pre {{ margin: 0; overflow-x: auto; white-space: pre-wrap; word-wrap: break-word; }}
        code {{ font-family: Consolas, Monaco, 'Andale Mono', 'Ubuntu Mono', monospace; font-size: 14px; line-height: 1.5; }}
    </style>
</head>
<body>
    <div class="dashboard-container">
        {plot_div}
        <div class="md-container">
            <button class="copy-btn" onclick="copyMarkdown()">Copy Markdown</button>
            <pre><code id="md-content">{markdown_text}</code></pre>
        </div>
    </div>
    <script>
        function copyMarkdown() {{
            var content = document.getElementById('md-content').innerText;
            navigator.clipboard.writeText(content).then(function() {{
                var btn = document.querySelector('.copy-btn');
                var originalText = btn.innerText;
                btn.innerText = 'Copied!';
                setTimeout(function() {{ btn.innerText = originalText; }}, 2000);
            }});
        }}
    </script>
</body>
</html>
"""


def plot_recommender_dashboard(run_config, metrics, output_dir="results"):
    """Plots Training Loss and Test Accuracy for the Recommender LSTM and generates unified HTML."""
    os.makedirs(output_dir, exist_ok=True)

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'[{run_config.run}] Training Loss over Epochs',
                        f'[{run_config.run}] Test Accuracy over Epochs'),
        vertical_spacing=0.15
    )

    fig.add_trace(go.Scatter(x=metrics.epochs, y=metrics.train_loss,
                  mode='lines+markers', name='Loss', line=dict(color='red')), row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=1)

    fig.add_trace(go.Scatter(x=metrics.epochs, y=metrics.test_accuracy,
                  mode='lines+markers', name='Accuracy', line=dict(color='green')), row=2, col=1)
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_yaxes(title_text="Accuracy", row=2, col=1)

    fig.update_layout(height=800, width=1000,
                      title_text=f"Run {run_config.run} - Dashboard")

    # Generate HTML string for Plotly
    plot_div = fig.to_html(full_html=False, include_plotlyjs='cdn')

    # Generate Markdown String
    md_text = get_markdown_text(run_config, metrics)

    # Build complete custom HTML
    html_content = build_html_template(
        plot_div, md_text, f"Run {run_config.run} Dashboard")

    # Save the files
    with open(Path(output_dir) / f"{run_config.run}_dashboard.html", "w") as f:
        f.write(html_content)

    fig.write_image(Path(output_dir) / f"{run_config.run}_dashboard.png")


def plot_forecasting_dashboard(run_config, dates, data, test_dates, test_preds, future_dates, future_preds, metrics, output_dir="results"):
    """Plots the actual timeline vs predictions and future inference and generates unified HTML."""
    os.makedirs(output_dir, exist_ok=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=data, mode='lines',
                  name='Actual Demand', line=dict(color='gray')))
    fig.add_trace(go.Scatter(x=test_dates, y=test_preds, mode='lines',
                  name=f'{run_config.run} Test Preds', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=future_dates, y=future_preds, mode='lines',
                  name=f'{run_config.run} Future', line=dict(color='red', dash='dash')))

    fig.update_layout(height=600, width=1200,
                      title_text=f"Run {run_config.run} - Demand Forecasting Inference", xaxis_title="Date", yaxis_title="Volume")

    # Generate HTML string for Plotly
    plot_div = fig.to_html(full_html=False, include_plotlyjs='cdn')

    # Generate Markdown String
    md_text = get_markdown_text(run_config, metrics)

    # Build complete custom HTML
    html_content = build_html_template(
        plot_div, md_text, f"Run {run_config.run} Dashboard")

    # Save the files
    with open(Path(output_dir) / f"{run_config.run}_dashboard.html", "w") as f:
        f.write(html_content)

    fig.write_image(Path(output_dir) / f"{run_config.run}_dashboard.png")


def plot_ncf_dashboard(run_config, metrics, top_k_df, history_df, output_dir="results"):
    """Plots Training Loss, Val Accuracy, History, and Top-K for NCF."""
    os.makedirs(output_dir, exist_ok=True)

    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(f'[{run_config.run}] Training Loss',
                        f'[{run_config.run}] Val Accuracy',
                        'User Purchase History (Last 5 Items)',
                        'Top-K Recommendations (Predicted Probability)'),
        specs=[[{"type": "xy"}], [{"type": "xy"}], [{"type": "table"}], [{"type": "xy"}]],
        vertical_spacing=0.08,
        row_heights=[0.15, 0.15, 0.25, 0.45]
    )

    # --- 1. Loss ---
    fig.add_trace(go.Scatter(x=metrics.epochs, y=metrics.train_loss,
                  mode='lines+markers', name='Loss', line=dict(color='red')), row=1, col=1)

    # --- 2. Accuracy ---
    fig.add_trace(go.Scatter(x=metrics.epochs, y=metrics.test_accuracy,
                  mode='lines+markers', name='Accuracy', line=dict(color='green')), row=2, col=1)

    # --- 3. History Table ---
    fig.add_trace(
        go.Table(
            header=dict(values=["User ID", "Previously Purchased Product"],
                        fill_color='indigo', font=dict(color='white'), align='left'),
            cells=dict(values=[history_df.user_id_orig, history_df.product_name],
                       fill_color='darkslateblue', font=dict(color='white'), align='left')
        ),
        row=3, col=1
    )

    # --- 4. Grouped Horizontal Bar Chart ---
    # Sort for display: Rank 1 at the top for each user
    plot_df = top_k_df.sort_values(['user_id_orig', 'rank'], ascending=[True, False])
    
    for user_id in plot_df['user_id_orig'].unique():
        user_data = plot_df[plot_df['user_id_orig'] == user_id]
        fig.add_trace(
            go.Bar(
                y=user_data['product_name'],
                x=user_data['score'],
                name=f"User {user_id}",
                orientation='h',
                text=user_data['score'].apply(lambda x: f"{x:.4f}"),
                textposition='auto',
                hovertemplate="User: " + user_id + "<br>Product: %{y}<br>Prob: %{x:.4f}<extra></extra>"
            ),
            row=4, col=1
        )

    fig.update_layout(height=1600, width=1100,
                      template="plotly_dark",
                      title_text=f"NCF Run {run_config.run} - Comprehensive Dashboard",
                      barmode='group',
                      legend_title="Users")
    
    fig.update_yaxes(title_text="Product Name", row=4, col=1)
    fig.update_xaxes(title_text="Predicted Probability", row=4, col=1)


    # Generate HTML string for Plotly
    plot_div = fig.to_html(full_html=False, include_plotlyjs='cdn')

    # Generate Markdown String
    md_text = get_markdown_text(run_config, metrics)

    # Build complete custom HTML
    html_content = build_html_template(
        plot_div, md_text, f"Run {run_config.run} Dashboard")

    # Save the files
    with open(Path(output_dir) / f"{run_config.run}_dashboard.html", "w") as f:
        f.write(html_content)

    fig.write_image(Path(output_dir) / f"{run_config.run}_dashboard.png")
