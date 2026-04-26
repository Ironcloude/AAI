import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import time

import run_utils
from config import get_run_config, ForecastingMetrics


def prepare_data_for_xgb(filepath='groceries_dataset.csv', freq='D'):
    """
    Loads and prepares the dataset specifically for XGBoost.
    Unlike LSTM/SARIMAX, XGBoost requires explicitly engineered tabular features 
    (lags, rolling stats, date parts) instead of raw sequential data.
    """
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return None, None, None, None

    # 1. Load and aggregate to true chronological series
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df = df.sort_values(by='Date')

    # Handle aggregation based on frequency config
    if freq in ['W', 'ME', 'M']:
        agg = df.groupby(pd.Grouper(key='Date', freq=freq)
                         ).size().reset_index(name='demand')
    else:
        agg = df.groupby('Date').size().reset_index(name='demand')

    agg.set_index('Date', inplace=True)

    full_idx = pd.date_range(start=agg.index.min(),
                             end=agg.index.max(), freq=freq)
    agg = agg.reindex(full_idx, fill_value=0)

    df_ts = agg.copy()

    # 2. Feature Engineering
    # Date parts
    df_ts['dayofweek'] = df_ts.index.dayofweek
    df_ts['month'] = df_ts.index.month
    df_ts['dayofmonth'] = df_ts.index.day
    df_ts['is_weekend'] = df_ts['dayofweek'].isin([5, 6]).astype(int)

    # Dynamically adjust lags and rolling windows based on frequency
    if freq in ['ME', 'M']:
        lags = [1, 2, 3, 6, 12]  # Months
        window_short, window_long = 3, 6
    else:
        lags = [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]  # Days
        window_short, window_long = 7, 14

    for lag in lags:
        df_ts[f'lag_{lag}'] = df_ts['demand'].shift(lag)

    # Rolling features (moving averages and volatility)
    df_ts[f'rolling_mean_{window_short}'] = df_ts['demand'].shift(
        1).rolling(window=window_short).mean()
    df_ts[f'rolling_std_{window_short}'] = df_ts['demand'].shift(
        1).rolling(window=window_short).std()
    df_ts[f'rolling_mean_{window_long}'] = df_ts['demand'].shift(
        1).rolling(window=window_long).mean()

    # Drop rows which now contain NaNs due to the longest lag shift
    df_ts.dropna(inplace=True)

    return df_ts, lags, window_short, window_long


def run_xgboost_forecaster(run_id="C3", output_dir="forecasting_results"):
    print(f"\n--- Starting Standalone XGBoost Forecasting: {run_id} ---")
    start_time = time.time()

    run_config = get_run_config(run_id)
    if not run_config:
        print(f"Run config {run_id} not found.")
        return

    df, lags, window_short, window_long = prepare_data_for_xgb(
        freq=run_config.freq)
    if df is None:
        return

    # Split temporally (80% train, 20% test)
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    features = [col for col in df.columns if col != 'demand']
    target = 'demand'

    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]

    unit = "months" if run_config.freq in ['ME', 'M'] else "days"
    print(f"Engineered {len(features)} features.")
    print(f"Training on {len(X_train)} {unit}, testing on {len(X_test)} {unit}.")

    model = xgb.XGBRegressor(
        n_estimators=run_config.n_estimators,
        learning_rate=run_config.learning_rate,
        max_depth=run_config.max_depth,
        subsample=run_config.subsample,
        colsample_bytree=run_config.colsample_bytree,
        objective='reg:squarederror',
        random_state=42,
        early_stopping_rounds=run_config.early_stopping_rounds
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False  # Set to False for cleaner console like A/B models
    )

    # Predict Test Set
    test_preds = model.predict(X_test)

    # Populate Metrics (matching config structure)
    metrics = ForecastingMetrics(run=f"{run_config.run}_metrics")
    metrics.mae = float(mean_absolute_error(y_test, test_preds))
    metrics.rmse = float(np.sqrt(mean_squared_error(y_test, test_preds)))
    metrics.elapsed_time_min = (time.time() - start_time) / 60

    # Autoregressive Future Prediction
    future_dates = pd.date_range(
        start=df.index[-1],
        periods=run_config.future_steps + 1,
        freq=run_config.freq
    )[1:]

    future_preds = []
    # Keep a running history buffer for recursive feature generation
    history_demand = list(df['demand'].values)

    for i in range(run_config.future_steps):
        curr_date = future_dates[i]
        feat_dict = {}

        # Calendar features
        feat_dict['dayofweek'] = curr_date.dayofweek
        feat_dict['month'] = curr_date.month
        feat_dict['dayofmonth'] = curr_date.day
        feat_dict['is_weekend'] = int(curr_date.dayofweek in [5, 6])

        # Lags
        for lag in lags:
            feat_dict[f'lag_{lag}'] = history_demand[-lag]

        # Rolling stats
        feat_dict[f'rolling_mean_{window_short}'] = np.mean(
            history_demand[-window_short:])
        feat_dict[f'rolling_std_{window_short}'] = np.std(
            history_demand[-window_short:], ddof=1) if len(history_demand) > 1 else 0
        feat_dict[f'rolling_mean_{window_long}'] = np.mean(
            history_demand[-window_long:])

        # Enforce exact column order
        curr_X = pd.DataFrame([feat_dict])[features]

        # Predict and update buffer
        pred = model.predict(curr_X)[0]
        future_preds.append(float(pred))
        history_demand.append(float(pred))

    print(f"\n--- Final Results ---")
    print(f"Test MAE:  {metrics.mae:.4f}")
    print(f"Test RMSE: {metrics.rmse:.4f}")

    # Use run_utils to save JSON and generate exact same HTML Plotly dashboards
    run_utils.save_run_data(run_config, metrics, output_dir)
    run_utils.plot_forecasting_dashboard(
        run_config=run_config,
        dates=df.index,
        data=df['demand'].values,
        test_dates=test_df.index,
        test_preds=test_preds,
        future_dates=future_dates,
        future_preds=future_preds,
        metrics=metrics,
        output_dir=output_dir
    )

    # Export XGBoost Model safely
    model.save_model(os.path.join(output_dir, f'{run_config.run}_model.json'))


if __name__ == "__main__":
    run_xgboost_forecaster("C1")
    run_xgboost_forecaster("C2")
    run_xgboost_forecaster("C3")
    run_xgboost_forecaster("C4")
