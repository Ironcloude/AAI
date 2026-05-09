import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from safetensors.torch import save_file

from config import get_run_config, ForecastingMetrics
import run_utils


def prepare_true_timeseries(filepath='groceries_dataset.csv', freq='D'):
    """
    Builds a TRUE chronological time-series from raw timestamp data.
    By default, calculates total daily grocery demand. If freq='W', 
    calculates total weekly grocery demand to reduce noise.
    """
    df = pd.read_csv(filepath)

    # Convert to datetime and sort chronologically
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df = df.sort_values(by='Date')

    # Handle aggregation based on frequency config
    if freq in ['W', 'ME', 'M']:
        aggregated_demand = df.groupby(pd.Grouper(
            key='Date', freq=freq)).size().reset_index(name='demand')
    else:
        # Default Daily aggregation
        aggregated_demand = df.groupby(
            'Date').size().reset_index(name='demand')

    aggregated_demand.set_index('Date', inplace=True)

    # Reindex missing dates/weeks with zero demand to maintain strict chronology
    full_date_range = pd.date_range(
        start=aggregated_demand.index.min(), end=aggregated_demand.index.max(), freq=freq)
    aggregated_demand = aggregated_demand.reindex(
        full_date_range, fill_value=0)

    return aggregated_demand['demand'].values.astype(float), aggregated_demand.index


def create_lstm_sequences(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:(i + lookback)])
        y.append(data[i + lookback])
    return np.array(X), np.array(y)


class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        super(TimeSeriesLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        out = self.fc1(last_out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def run_sarimax(run_id="B1", output_dir="forecasting_results"):
    run_config = get_run_config(run_id)
    if not run_config:
        return

    print(f"\n--- Starting SARIMAX Run: {run_config.run} ---")
    start_time = time.time()

    # Reason: monthly ACF/PACF is cleaner than weekly -> better forecasting signal
    data, dates = prepare_true_timeseries(
        freq=getattr(run_config, 'freq', 'D'))
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]

    model = SARIMAX(train_data, order=run_config.order, seasonal_order=run_config.seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    # Increase maxiter to help B2 converge on noisy daily data
    fitted_model = model.fit(disp=False, maxiter=200)

    os.makedirs(output_dir, exist_ok=True)
    # NOTE: SARIMAX relies on statsmodels objects; cannot use .safetensors
    fitted_model.save(os.path.join(output_dir, f'{run_config.run}.pkl'))

    full_forecast = fitted_model.forecast(
        steps=len(test_data) + run_config.future_steps)
    test_preds = full_forecast[:len(test_data)]
    future_preds = full_forecast[len(test_data):]

    metrics = ForecastingMetrics(run=f"{run_config.run}_metrics")
    metrics.mae = mean_absolute_error(test_data, test_preds)
    metrics.rmse = np.sqrt(mean_squared_error(test_data, test_preds))
    metrics.elapsed_time_min = (time.time() - start_time) / 60

    run_utils.save_run_data(run_config, metrics, output_dir)

    test_dates = dates[train_size:]
    # Update future_dates to use the appropriate frequency (monthly, weekly, daily)
    future_dates = pd.date_range(
        start=dates[-1], periods=run_config.future_steps + 1, freq=getattr(run_config, 'freq', 'D'))[1:]
    run_utils.plot_forecasting_dashboard(
        run_config, dates, data, test_dates, test_preds, future_dates, future_preds, metrics, output_dir)


def run_forecasting_lstm(run_id="A1", output_dir="forecasting_results"):
    run_config = get_run_config(run_id)
    if not run_config:
        return

    print(f"\n--- Starting Forecasting LSTM Run: {run_config.run} ---")
    start_time = time.time()

    # Reason: monthly ACF/PACF is cleaner than weekly -> better forecasting signal
    data, dates = prepare_true_timeseries(
        freq=getattr(run_config, 'freq', 'D'))
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data.reshape(-1, 1)).flatten()
    test_scaled = scaler.transform(test_data.reshape(-1, 1)).flatten()

    X_train, y_train = create_lstm_sequences(
        train_scaled, run_config.lookback_window)
    X_test, y_test = create_lstm_sequences(
        test_scaled, run_config.lookback_window)

    if len(X_train) == 0 or len(X_test) == 0:
        print(f"Error: Not enough data for lookback_window={run_config.lookback_window} at freq={getattr(run_config, 'freq', 'D')}")
        return

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(
        y_train, dtype=torch.float32).unsqueeze(1)), batch_size=run_config.batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(
        y_test, dtype=torch.float32).unsqueeze(1)), batch_size=run_config.batch_size, shuffle=False)

    model = TimeSeriesLSTM(hidden_size=run_config.hidden_size,
                           num_layers=run_config.num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=run_config.learning_rate)

    loss_type = getattr(run_config, 'loss_type', 'MSE')
    if loss_type == "Huber":
        criterion = nn.HuberLoss()
    elif loss_type == "MAE":
        criterion = nn.L1Loss()
    else:
        criterion = nn.MSELoss()

    metrics = ForecastingMetrics(run=f"{run_config.run}_metrics")

    for epoch in range(run_config.epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        metrics.train_loss.append(total_loss / len(train_loader))

    os.makedirs(output_dir, exist_ok=True)
    save_file(model.state_dict(), os.path.join(
        output_dir, f'{run_config.run}.safetensors'))

    model.eval()
    predictions_scaled = []
    with torch.no_grad():
        for batch_X, _ in test_loader:
            predictions_scaled.extend(
                model(batch_X.to(device)).cpu().numpy().flatten())

    predictions = scaler.inverse_transform(
        np.array(predictions_scaled).reshape(-1, 1)).flatten()
    y_test_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    metrics.mae = mean_absolute_error(y_test_true, predictions)
    metrics.rmse = np.sqrt(mean_squared_error(y_test_true, predictions))
    metrics.elapsed_time_min = (time.time() - start_time) / 60

    # Future Inference
    last_seq = test_scaled[-run_config.lookback_window:
                           ].reshape(1, run_config.lookback_window, 1)
    future_preds_scaled = []
    with torch.no_grad():
        curr_seq = torch.tensor(last_seq, dtype=torch.float32).to(device)
        for _ in range(run_config.future_steps):
            pred = model(curr_seq)
            future_preds_scaled.append(pred.item())
            curr_seq = torch.cat(
                [curr_seq[:, 1:, :], pred.unsqueeze(1)], dim=1)

    future_forecast = scaler.inverse_transform(
        np.array(future_preds_scaled).reshape(-1, 1)).flatten()

    run_utils.save_run_data(run_config, metrics, output_dir)

    test_dates = dates[train_size + run_config.lookback_window:]
    future_dates = pd.date_range(
        start=dates[-1], periods=run_config.future_steps + 1, freq=getattr(run_config, 'freq', 'D'))[1:]
    run_utils.plot_forecasting_dashboard(
        run_config, dates, data, test_dates, predictions, future_dates, future_forecast, metrics, output_dir)


if __name__ == '__main__':
    # 1-2 Daily Runs (Baseline)
    # 3-4 Monthly Runs (Optimized Seasonality)
    run_forecasting_lstm("A1")
    run_forecasting_lstm("A2")
    run_forecasting_lstm("A3")
    run_forecasting_lstm("A4")

    run_sarimax("B1")
    run_sarimax("B2")
    run_sarimax("B3")
    run_sarimax("B4")
