import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass, field
from typing import List
from safetensors.torch import save_file

import run_utils
from config import get_run_config, RecommenderMetrics


@dataclass
class FMConfig:
    run: str = "F1"
    embed_dim: int = 32
    learning_rate: float = 0.001
    batch_size: int = 4096
    epochs: int = 1
    sample_frac: float = 0.1
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

    cat_features = ['user_id', 'product_id']

    # Encode IDs to match NCF structure for Top-K logic
    user_map = {u: i for i, u in enumerate(df['user_id'].unique())}
    item_map = {p: i for i, p in enumerate(df['product_id'].unique())}
    product_name_map = df[['product_id', 'product_name']].drop_duplicates().set_index('product_id')['product_name'].to_dict()
    
    for d in [train_df, val_df, test_df]:
        d['user_id'] = d['user_id'].map(user_map)
        d['product_id'] = d['product_id'].map(item_map)

    field_dims = [len(user_map), len(item_map)]

    # Extract arrays
    X_train = train_df[cat_features].values
    y_train = train_df['is_reorder'].values

    X_val = val_df[cat_features].values
    y_val = val_df['is_reorder'].values

    X_test = test_df[cat_features].values
    y_test = test_df['is_reorder'].values

    return (X_train, y_train, X_val, y_val, X_test, y_test, 
            field_dims, len(user_map), len(item_map), 
            test_df, user_map, item_map, product_name_map, train_df)


def calculate_precision_recall_at_k(model, u_test, i_test, y_test, num_items, device, k=10):
    model.eval()
    user_test_data = pd.DataFrame({
        'user_id': u_test,
        'product_id': i_test,
        'label': y_test
    })

    positive_users = user_test_data[user_test_data['label'] == 1]['user_id'].unique()
    if len(positive_users) == 0:
        return 0.0, 0.0

    precisions = []
    recalls = []

    sampled_users = np.random.choice(positive_users, min(100, len(positive_users)), replace=False)

    for u_idx in sampled_users:
        actual_relevant = set(user_test_data[(user_test_data['user_id'] == u_idx) & (
            user_test_data['label'] == 1)]['product_id'].values)

        x_vec = torch.stack([
            torch.full((num_items,), u_idx, dtype=torch.long),
            torch.arange(num_items, dtype=torch.long)
        ], dim=1).to(device)

        with torch.no_grad():
            scores = model(x_vec).cpu().numpy()

        top_k_idx = np.argsort(scores)[-k:][::-1]
        top_k_items = set(top_k_idx)

        hits = len(actual_relevant.intersection(top_k_items))
        precisions.append(hits / k)
        recalls.append(hits / len(actual_relevant))

    return np.mean(precisions), np.mean(recalls)


def predict_random_items(model, user_idx, num_items, device, k=5):
    model.eval()
    item_indices = np.random.choice(num_items, size=k, replace=False)

    x_vec = torch.stack([
        torch.full((k,), user_idx, dtype=torch.long),
        torch.tensor(item_indices, dtype=torch.long)
    ], dim=1).to(device)

    with torch.no_grad():
        scores = model(x_vec).cpu().numpy()

    return item_indices, scores


def train_fm(run_id="D1", output_dir="recommender_results"):
    config = FMConfig(run=run_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (X_train, y_train, X_val, y_val, X_test, y_test, 
     field_dims, num_users, num_items, 
     test_df, user_map, item_map, product_name_map, train_df) = load_data(sample_frac=config.sample_frac)

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
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Acc: {final_val_acc:.4f}")

    metrics.elapsed_time_min = (time.time() - start_time) / 60
    metrics.final_accuracy = metrics.test_accuracy[-1]

    os.makedirs(output_dir, exist_ok=True)
    save_file(model.state_dict(), os.path.join(output_dir, f'{config.run}.safetensors'))

    print("--- Calculating Precision@K and Recall@K...")
    u_test = X_test[:, 0]
    i_test = X_test[:, 1]
    p_at_k, r_at_k = calculate_precision_recall_at_k(
        model, u_test, i_test, y_test, num_items, device, k=10
    )
    metrics.precision_at_k = p_at_k
    metrics.recall_at_k = r_at_k
    print(f"Precision@10: {p_at_k:.4f} | Recall@10: {r_at_k:.4f}")

    run_utils.save_run_data(config, metrics, output_dir)

    # TOP-K & HISTORY SAMPLE FOR DASHBOARD
    model.eval()
    with torch.no_grad():
        unique_test_users = test_df['user_id'].unique()
        sample_user_ids = [np.random.choice(unique_test_users)]

        inv_user_map = {i: u for u, i in user_map.items()}
        inv_item_map = {i: p for p, i in item_map.items()}

        u_idx = sample_user_ids[0]
        items, scores = predict_random_items(model, u_idx, num_items, device, k=5)
        for i, s in zip(items, scores):
            pid = inv_item_map[i]
            name = product_name_map.get(pid, str(pid))
            print(f"{name} → {s:.4f}")

        top_k_list = []
        history_list = []
        k = 10

        for u_idx in sample_user_ids:
            # 1. History
            u_hist = train_df[train_df['user_id'] == u_idx].sort_values('order_number', ascending=False).head(5)
            for _, row in u_hist.iterrows():
                history_list.append({
                    'user_id_orig': str(inv_user_map[u_idx]),
                    'product_name': product_name_map.get(inv_item_map[row['product_id']], "Unknown")
                })

            # 2. Top-K Recommendations
            x_vec = torch.stack([
                torch.full((num_items,), u_idx, dtype=torch.long),
                torch.arange(num_items, dtype=torch.long)
            ], dim=1).to(device)
            scores = model(x_vec).cpu().numpy()

            top_indices = np.argsort(scores)[-k:][::-1]
            for rank, i_idx in enumerate(top_indices):
                top_k_list.append({
                    'user_id': u_idx,
                    'user_id_orig': str(inv_user_map[u_idx]),
                    'product_id': i_idx,
                    'product_name': product_name_map.get(inv_item_map[i_idx], "Unknown"),
                    'score': scores[i_idx],
                    'rank': rank + 1
                })

        top_k_df = pd.DataFrame(top_k_list)
        history_df = pd.DataFrame(history_list)

    # DASHBOARD
    run_utils.plot_ncf_dashboard(config, metrics, top_k_df, history_df, output_dir)

    print(f"FM training complete: {config.run}")

if __name__ == "__main__":
    train_fm("F1")
