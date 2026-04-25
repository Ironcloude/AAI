import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from safetensors.torch import save_file

import run_utils
from config import RecommenderMetrics


# CONFIG
@dataclass
class NCFConfig:
    run: str = "E1"
    embed_dim: int = 32
    hidden_dims: list = None
    learning_rate: float = 0.001
    batch_size: int = 4096
    epochs: int = 1
    sample_frac: float = 0.1
    model_type: str = "recommender_ncf"

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 32]


# MODEL
class NCF(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, hidden_dims):
        super().__init__()

        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)

        layers = []
        input_dim = embed_dim * 2

        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h

        layers.append(nn.Linear(input_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, user, item):
        u = self.user_embed(user)
        i = self.item_embed(item)

        x = torch.cat([u, i], dim=1)
        return torch.sigmoid(self.mlp(x)).view(-1)


# DATA LOADER
def load_data(filepath='insta_clean_data.parquet', sample_frac=1.0):
    print(f"--- Loading data (NCF): {filepath}, frac={sample_frac}")
    df = pd.read_parquet(filepath)

    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)

    df['days_since_last_order'] = df['days_since_last_order'].fillna(0)

    # Split
    train_df = df[df['eval_set'] == 'prior'].copy()
    test_df = df[df['eval_set'] == 'train'].copy()

    # Validation split
    train_df = train_df.sort_values(['user_id', 'order_number'])
    val_idx = train_df.groupby('user_id')['order_number'].transform(
        lambda x: x >= x.max() - 1
    )

    val_df = train_df[val_idx].copy()
    train_df = train_df[~val_idx].copy()

    # Encode IDs
    user_map = {u: i for i, u in enumerate(df['user_id'].unique())}
    item_map = {p: i for i, p in enumerate(df['product_id'].unique())}
    product_name_map = df[['product_id', 'product_name']].drop_duplicates(
    ).set_index('product_id')['product_name'].to_dict()

    for d in [train_df, val_df, test_df]:
        d['user_id'] = d['user_id'].map(user_map)
        d['product_id'] = d['product_id'].map(item_map)

    def to_xy(data):
        X_user = data['user_id'].values
        X_item = data['product_id'].values
        y = data['is_reorder'].values
        return X_user, X_item, y

    return (
        *to_xy(train_df),
        *to_xy(val_df),
        *to_xy(test_df),
        len(user_map),
        len(item_map),
        test_df,
        user_map,
        item_map,
        product_name_map,
        train_df
    )


# METRICS
def calculate_precision_recall_at_k(model, u_test, i_test, y_test, num_items, device, k=10):
    model.eval()
    user_test_data = pd.DataFrame({
        'user_id': u_test,
        'product_id': i_test,
        'label': y_test
    })

    # We only care about users who have at least one positive label in test set
    positive_users = user_test_data[user_test_data['label']
                                    == 1]['user_id'].unique()
    if len(positive_users) == 0:
        return 0.0, 0.0

    precisions = []
    recalls = []

    # For efficiency, we sample some users if the test set is large
    sampled_users = np.random.choice(positive_users, min(
        100, len(positive_users)), replace=False)

    for u_idx in sampled_users:
        # Get ground truth items (relevant items)
        actual_relevant = set(user_test_data[(user_test_data['user_id'] == u_idx) & (
            user_test_data['label'] == 1)]['product_id'].values)

        # Predict for all products
        user_tensor = torch.full(
            (num_items,), u_idx, dtype=torch.long).to(device)
        item_tensor = torch.arange(num_items, dtype=torch.long).to(device)

        with torch.no_grad():
            scores = model(user_tensor, item_tensor).cpu().numpy()

        top_k_idx = np.argsort(scores)[-k:][::-1]
        top_k_items = set(top_k_idx)

        hits = len(actual_relevant.intersection(top_k_items))
        precisions.append(hits / k)
        recalls.append(hits / len(actual_relevant))

    return np.mean(precisions), np.mean(recalls)


def predict_random_items(model, user_idx, num_items, device, k=5):
    model.eval()

    # pick random items
    item_indices = np.random.choice(num_items, size=k, replace=False)

    user_tensor = torch.full((k,), user_idx, dtype=torch.long).to(device)
    item_tensor = torch.tensor(item_indices, dtype=torch.long).to(device)

    with torch.no_grad():
        scores = model(user_tensor, item_tensor).cpu().numpy()

    return item_indices, scores

# TRAINING


def train_ncf(run_id="D2", output_dir="recommender_results"):
    config = NCFConfig(run=run_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (
        u_train, i_train, y_train,
        u_val, i_val, y_val,
        u_test, i_test, y_test,
        num_users, num_items,
        test_df,
        user_map,
        item_map,
        product_name_map,
        train_df
    ) = load_data(sample_frac=config.sample_frac)

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(u_train, dtype=torch.long),
            torch.tensor(i_train, dtype=torch.long),
            torch.tensor(y_train, dtype=torch.float32)
        ),
        batch_size=config.batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(u_val, dtype=torch.long),
            torch.tensor(i_val, dtype=torch.long),
            torch.tensor(y_val, dtype=torch.float32)
        ),
        batch_size=config.batch_size,
        shuffle=False
    )

    model = NCF(num_users, num_items,
                config.embed_dim, config.hidden_dims).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCELoss()

    metrics = RecommenderMetrics(run=config.run)

    print(f"--- Training NCF {config.run}")
    start_time = time.time()

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0

        for u, i, y in train_loader:
            u, i, y = u.to(device), i.to(device), y.to(device)

            optimizer.zero_grad()
            preds = model(u, i)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for u, i, y in val_loader:
                u, i, y = u.to(device), i.to(device), y.to(device)
                preds = model(u, i)
                correct += ((preds > 0.5) == y).sum().item()
                total += y.size(0)

        acc = correct / total
        avg_loss = total_loss / len(train_loader)

        metrics.epochs.append(epoch + 1)
        metrics.train_loss.append(avg_loss)
        metrics.test_accuracy.append(acc)

        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Val Acc: {acc:.4f}")

    metrics.final_accuracy = metrics.test_accuracy[-1]
    metrics.elapsed_time_min = (time.time() - start_time) / 60

    # Save model
    os.makedirs(output_dir, exist_ok=True)
    save_file(model.state_dict(),
              os.path.join(output_dir, f"{config.run}.safetensors"))

    # Calculate Precision@K and Recall@K
    print("--- Calculating Precision@K and Recall@K...")
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
        # Pick 1 user from test set
        # Pick 3 users from test set
        unique_test_users = test_df['user_id'].unique()

        # sample_user_ids = np.random.choice(unique_test_users, min(3, len(unique_test_users)), replace=False)
        sample_user_ids = [np.random.choice(unique_test_users)]

        # new
        # create mappings
        inv_user_map = {i: u for u, i in user_map.items()}
        inv_item_map = {i: p for p, i in item_map.items()}

        u_idx = sample_user_ids[0]  # your selected user
        items, scores = predict_random_items(
            model, u_idx, num_items, device, k=5)

        for i, s in zip(items, scores):
            pid = inv_item_map[i]
            name = product_name_map.get(pid, str(pid))
            print(f"{name} → {s:.4f}")
        # new

        top_k_list = []
        history_list = []
        k = 10

        inv_user_map = {i: u for u, i in user_map.items()}
        inv_item_map = {i: p for p, i in item_map.items()}

        for u_idx in sample_user_ids:
            # 1. History
            # Extract last 5 from train_df
            u_hist = train_df[train_df['user_id'] == u_idx].sort_values(
                'order_number', ascending=False).head(5)
            for _, row in u_hist.iterrows():
                history_list.append({
                    'user_id_orig': str(inv_user_map[u_idx]),
                    'product_name': product_name_map.get(inv_item_map[row['product_id']], "Unknown")
                })

            # 2. Top-K Recommendations
            user_tensor = torch.full(
                (num_items,), u_idx, dtype=torch.long).to(device)
            item_tensor = torch.arange(num_items, dtype=torch.long).to(device)
            scores = model(user_tensor, item_tensor).cpu().numpy()

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
    run_utils.plot_ncf_dashboard(
        config, metrics, top_k_df, history_df, output_dir)

    print(f"NCF training complete: {config.run}")


if __name__ == "__main__":
    train_ncf("E1")
