import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from safetensors.torch import save_file

from config import get_run_config, RecommenderMetrics
import run_utils


# Pad user sequences to a fixed length for batching.
def pad_sequences_custom(sequences, maxlen, padding='pre'):
    padded = np.zeros((len(sequences), maxlen), dtype=int)
    for i, seq in enumerate(sequences):
        seq_len = len(seq)
        if seq_len > maxlen:
            if padding == 'pre':
                padded[i] = seq[-maxlen:]
            else:
                padded[i] = seq[:maxlen]
        else:
            if padding == 'pre':
                padded[i, maxlen-seq_len:] = seq
            else:
                padded[i, :seq_len] = seq
    return padded


def load_and_prepare_sequences(filepath='insta_clean_data.csv', min_seq_length=3, max_lookback=20):
    if not os.path.exists(filepath):
        print(f"File {filepath} not found.")
        return None, None, None, None

    df = pd.read_csv(filepath)
    if 'cart_position' in df.columns:
        df = df.sort_values(by=['user_id', 'order_id', 'cart_position'])
    else:
        df = df.sort_values(by=['user_id', 'order_id'])

    label_encoder = LabelEncoder()
    df['product_id_encoded'] = label_encoder.fit_transform(df['product_name'])
    vocab_size = len(label_encoder.classes_) + 1
    df['product_id_encoded'] = df['product_id_encoded'] + 1

    user_sequences = df.groupby(
        'user_id')['product_id_encoded'].apply(list).values
    user_sequences = [
        seq for seq in user_sequences if len(seq) >= min_seq_length]

    X, y = [], []
    for seq in user_sequences:
        for i in range(1, len(seq)):
            X.append(seq[:i])
            y.append(seq[i])

    X_padded = pad_sequences_custom(X, maxlen=max_lookback, padding='pre')
    return X_padded, np.array(y), vocab_size, label_encoder


# Define the LSTM model used for next-item recommendation.
class RecommendationLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):
        super(RecommendationLSTM, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embed_dim,
                            hidden_size=hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        last_out = lstm_out[:, -1, :]
        out = self.fc1(last_out)
        out = self.relu(out)
        return self.fc2(out)


# Train and evaluate the recommender model.
def train_recommender(run_id="E1", output_dir="recommender_results"):
    run_config = get_run_config(run_id)
    if not run_config:
        print(f"Run config {run_id} not found.")
        return

    print(f"\n--- Starting Run: {run_config.run} ---")
    start_time = time.time()

    X, y, vocab_size, label_encoder = load_and_prepare_sequences(
        max_lookback=run_config.max_lookback)
    if X is None:
        return

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = TensorDataset(torch.tensor(
        X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(
        X_test, dtype=torch.long), torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(
        train_dataset, batch_size=run_config.batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=run_config.batch_size, shuffle=False)

    model = RecommendationLSTM(
        vocab_size, run_config.embed_dim, run_config.hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=run_config.learning_rate)

    metrics = RecommenderMetrics(run=f"{run_config.run}_metrics")

    for epoch in range(run_config.epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        metrics.epochs.append(epoch + 1)
        metrics.train_loss.append(avg_loss)

        # Eval Test
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                logits = model(batch_X)
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == batch_y).sum().item()
                total += batch_y.size(0)

        acc = correct / total
        metrics.test_accuracy.append(acc)
        metrics.final_accuracy = acc
        print(
            f"Epoch {epoch+1}/{run_config.epochs} | Loss: {avg_loss:.4f} | Test Acc: {acc:.4f}")

    metrics.elapsed_time_min = (time.time() - start_time) / 60

    # Exports
    os.makedirs(output_dir, exist_ok=True)
    save_file(model.state_dict(), os.path.join(
        output_dir, f'{run_config.run}.safetensors'))
    np.save(os.path.join(
        output_dir, f'{run_config.run}_label_classes.npy'), label_encoder.classes_)

    run_utils.save_run_data(run_config, metrics, output_dir)
    run_utils.plot_recommender_dashboard(run_config, metrics, output_dir)


if __name__ == '__main__':
    train_recommender("E1")

