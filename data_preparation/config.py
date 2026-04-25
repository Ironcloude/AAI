from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Any

# ---------------------------------------------------------
# Hyperparameter Configurations
# ---------------------------------------------------------


@dataclass
class ForecastingLSTMConfig:
    run: str
    hidden_size: int = 64
    num_layers: int = 1
    learning_rate: float = 0.001
    batch_size: int = 16
    epochs: int = 30
    lookback_window: int = 14
    future_steps: int = 14
    model_type: str = "forecasting_lstm"
    freq: str = "D"
    loss_type: str = "MSE"


@dataclass
class ForecastingSARIMAXConfig:
    run: str
    order: Tuple[int, int, int] = (1, 1, 1)
    seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 7)
    future_steps: int = 14
    model_type: str = "forecasting_sarimax"
    freq: str = "D"

@dataclass
class ForecastingXGBoostConfig:
    run: str
    n_estimators: int = 3000
    learning_rate: float = 0.01
    max_depth: int = 6
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    early_stopping_rounds: int = 100
    future_steps: int = 14
    freq: str = "D"
    model_type: str = "forecasting_xgboost"

@dataclass
class RecommenderLSTMConfig:
    run: str
    embed_dim: int = 64
    hidden_dim: int = 128
    learning_rate: float = 0.001
    batch_size: int = 256
    epochs: int = 5
    min_seq_length: int = 3
    max_lookback: int = 20
    model_type: str = "recommender_lstm"

@dataclass
class RecommenderNCFConfig:
    run: str = "D2"
    embed_dim: int = 32
    hidden_dims: list = field(default_factory=lambda: [64, 32])
    learning_rate: float = 0.001
    batch_size: int = 4096
    epochs: int = 3
    sample_frac: float = 0.1
    model_type: str = "recommender_ncf"

# ---------------------------------------------------------
# Metrics Tracking Configurations
# ---------------------------------------------------------


@dataclass
class ForecastingMetrics:
    run: str
    mae: float | None = None
    rmse: float | None = None
    elapsed_time_min: float | None = None
    train_loss: List[float] = field(default_factory=list)  # Only used for LSTM


@dataclass
class RecommenderMetrics:
    run: str
    epochs: List[int] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    test_accuracy: List[float] = field(default_factory=list)
    precision_at_k: float | None = None
    recall_at_k: float | None = None
    elapsed_time_min: float | None = None
    final_accuracy: float | None = None

# ---------------------------------------------------------
# Explicit Run Definitions
# ---------------------------------------------------------
# A/B/C = forecasting models
# A3/A4, B3/B4, and C4 use monthly aggregation (better seasonality)
# D reserved for future models
# E = recommender (unchanged, renamed)

# A-Series: Forecasting LSTM Runs (Forecasting + LSTM)
A1 = ForecastingLSTMConfig(run="A1", hidden_size=64, num_layers=1, lookback_window=14, epochs=30, freq="D")
A2 = ForecastingLSTMConfig(run="A2", hidden_size=128, num_layers=2, lookback_window=21, epochs=50, freq="D")
A3 = ForecastingLSTMConfig(run="A3", hidden_size=64, num_layers=1, lookback_window=2, future_steps=3, epochs=30, freq="ME", loss_type="Huber")
A4 = ForecastingLSTMConfig(run="A4", hidden_size=128, num_layers=2, lookback_window=3, future_steps=3, epochs=50, freq="ME", loss_type="Huber")

# B-Series: Forecasting SARIMAX Runs (Forecasting + SARIMAX)
B1 = ForecastingSARIMAXConfig(run="B1", order=(1, 1, 0), seasonal_order=(0, 1, 1, 7), freq="D")
B2 = ForecastingSARIMAXConfig(run="B2", order=(2, 1, 2), seasonal_order=(1, 1, 1, 7), freq="D")
B3 = ForecastingSARIMAXConfig(run="B3", order=(1, 1, 1), seasonal_order=(0, 0, 0, 0), freq="ME", future_steps=3)
B4 = ForecastingSARIMAXConfig(run="B4", order=(1, 1, 1), seasonal_order=(1, 0, 0, 6), freq="ME", future_steps=3)

# C-Series: Forecasting XGBoost Runs (Forecasting + XGBoost)
C1 = ForecastingXGBoostConfig(run="C1", max_depth=6, n_estimators=1000, freq="D", future_steps=14)
C2 = ForecastingXGBoostConfig(run="C2", max_depth=8, n_estimators=2000, freq="D", future_steps=14)
C3 = ForecastingXGBoostConfig(run="C3", max_depth=4, n_estimators=1000, freq="ME", future_steps=3)
C4 = ForecastingXGBoostConfig(run="C4", max_depth=6, n_estimators=2000, freq="ME", future_steps=3)

# D-Series: Recommender NCF Runs
D2 = RecommenderNCFConfig(run="D2", epochs=3)

# E-Series: Recommender LSTM Runs (Recommender + LSTM)
E1 = RecommenderLSTMConfig(run="E1", embed_dim=64,
                           hidden_dim=128, learning_rate=0.001, epochs=5)
E2 = RecommenderLSTMConfig(run="E2", embed_dim=128,
                           hidden_dim=256, learning_rate=0.005, epochs=10)


def get_run_config(run_id: str):
    return globals().get(run_id)
