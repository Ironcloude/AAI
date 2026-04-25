import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score
from dataclasses import dataclass, field
from typing import List
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from safetensors.torch import save_file

import run_utils
from config import get_run_config, RecommenderMetrics


@dataclass
class FMConfig:
    run: str = "D1"
    embed_dim: int = 16
    learning_rate: float = 0.001
    batch_size: int = 8192
    epochs: int = 1
    sample_frac: float = 0.1  # utilizing full dataset as requested
    model_type: str = "recommender_fm"


class FactorizationMachine(nn.Module):
    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]))

        self.linear = nn.Embedding(sum(field_dims), 1)
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)
        self.bias = nn.Parameter(torch.zeros(1))

        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, x):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)

        linear = torch.sum(self.linear(x), dim=1) + self.bias

        emb = self.embedding(x)
        square_of_sum = torch.sum(emb, dim=1) ** 2
        sum_of_square = torch.sum(emb ** 2, dim=1)

        interaction = 0.5 * \
            torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)

        return torch.sigmoid(linear + interaction).squeeze()


def load_data(filepath='insta_clean_data.parquet', sample_frac=1.0):
    print(f"--- Loading data (path: {filepath}, frac: {sample_frac})")
    df = pd.read_parquet(filepath)

    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)

    df['days_since_last_order'] = df['days_since_last_order'].fillna(0)

    # Split
    train_df = df[df['eval_set'] == 'prior'].copy()
    test_df = df[df['eval_set'] == 'train'].copy()

    # Time-aware validation split
    train_df = train_df.sort_values(['user_id', 'order_number'])
    val_idx = train_df.groupby('user_id')['order_number'].transform(
        lambda x: x >= x.max() - 1
    )

    val_df = train_df[val_idx].copy()
    train_df = train_df[~val_idx].copy()

    # Debug schema
    print("\n--- DATAFRAME SCHEMA")
    print(df.dtypes)

    cat_features = [
        'user_id',
        'product_id',
        'aisle_name',
        'department_name',
        'day_of_week',
        'hour_of_day'
    ]

    # Concatenate for consistent encoding
    train_len = len(train_df)
    val_len = len(val_df)

    full_df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)

    field_dims = []
    for col in cat_features:
        full_df[col], uniques = pd.factorize(full_df[col])
        field_dims.append(len(uniques))

    # Split back
    train_df = full_df.iloc[:train_len].copy()
    val_df = full_df.iloc[train_len:train_len + val_len].copy()
    test_df = full_df.iloc[train_len + val_len:].copy()

    # Extract arrays
    X_train = train_df[cat_features].values
    y_train = train_df['is_reorder'].values

    X_val = val_df[cat_features].values
    y_val = val_df['is_reorder'].values

    X_test = test_df[cat_features].values
    y_test = test_df['is_reorder'].values

    return X_train, y_train, X_val, y_val, X_test, y_test, field_dims, test_df


def train_fm(run_id="D1", output_dir="recommender_results"):
    config = FMConfig(run=run_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, y_train, X_val, y_val, X_test, y_test, field_dims, test_df = load_data(
        sample_frac=config.sample_frac)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.long),
                      torch.tensor(y_train, dtype=torch.float32)),
        batch_size=config.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.long),
                      torch.tensor(y_val, dtype=torch.float32)),
        batch_size=config.batch_size, shuffle=False
    )

    model = FactorizationMachine(field_dims, config.embed_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCELoss()

    # RecommenderMetrics from config.py
    metrics = RecommenderMetrics(run=f"{config.run}_metrics")
    start_time = time.time()

    print(f"--- Training FM Run {config.run}")
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        num_batches = len(train_loader)
        for i, (Xb, yb) in enumerate(train_loader):
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(Xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (i+1) % 50 == 0:
                print(
                    f"Epoch {epoch+1} | Batch {i+1}/{num_batches} | Loss: {loss.item():.4f}", end='\r')

        model.eval()
        val_acc = 0
        total_test = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                preds = model(Xb)
                val_acc += ((preds > 0.5) == yb).sum().item()
                total_test += yb.size(0)

        final_val_acc = val_acc / total_test
        avg_train_loss = total_loss / num_batches
        metrics.epochs.append(epoch + 1)
        metrics.train_loss.append(avg_train_loss)
        metrics.test_accuracy.append(final_val_acc)
        print(
            f"\nEpoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Acc: {final_val_acc:.4f}")

    metrics.elapsed_time_min = (time.time() - start_time) / 60
    metrics.final_accuracy = metrics.test_accuracy[-1]

    # Save Model & Exports
    os.makedirs(output_dir, exist_ok=True)
    # Save as Safetensors (Security First requirement)
    save_file(model.state_dict(), os.path.join(
        output_dir, f'{config.run}.safetensors'))

    # Save JSON data
    run_utils.save_run_data(config, metrics, output_dir)

    # Inference (correct, no encoders)
    print("\n--- Generating Inference Context")

    model.eval()
    test_user_ids = test_df['user_id'].unique()
    selected_users = np.random.choice(test_user_ids, 3, replace=False)

    inference_md = "\n### Behavioral Analysis (FM Context)\n"

    for uid in selected_users:
        user_rows = test_df[test_df['user_id'] == uid]
        if user_rows.empty:
            continue

        # pick random row index for that user
        # get positional index inside test_df
        row_pos = user_rows.sample(1).index[0]
        row_pos = test_df.index.get_loc(row_pos)

        x_vec = torch.tensor([X_test[row_pos]], dtype=torch.long).to(device)
        sample = test_df.iloc[row_pos]

        prob = model(x_vec).item()

        inference_md += (
            f"- User {sample['user_id']} | "
            f"Item: {sample['product_id']} | "
            f"Conf: {prob:.4f} | "
            f"Actual: {'YES' if sample['is_reorder'] else 'NO'}\n"
        )
    # generate html dashboard
    fig = make_subplots(rows=2, cols=1, subplot_titles=(
        'Training Loss (BCE)', 'Validation Accuracy'))
    fig.add_trace(go.Scatter(x=metrics.epochs, y=metrics.train_loss,
                  name='Loss', line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=metrics.epochs, y=metrics.test_accuracy,
                  name='Accuracy', line=dict(color='green')), row=2, col=1)

    fig.update_layout(height=800, template="plotly_dark",
                      title_text=f"Factorization Machine - {config.run}")
    plot_div = fig.to_html(full_html=False, include_plotlyjs='cdn')

    # get standard md report and append inference cases
    md_report = run_utils.get_markdown_text(config, metrics)
    md_report += inference_md

    html_dashboard = run_utils.build_html_template(
        plot_div, md_report, f"Run {config.run} Dashboard")

    with open(os.path.join(output_dir, f"{config.run}_dashboard.html"), "w", encoding="utf-8") as f:
        f.write(html_dashboard)

    print(f"Success! Model and Dashboard saved in {output_dir}")


if __name__ == "__main__":
    train_fm("D1")
