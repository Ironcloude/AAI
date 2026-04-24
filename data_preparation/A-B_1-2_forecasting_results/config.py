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

@dataclass
class ForecastingSARIMAXConfig:
    run: str
    order: Tuple[int, int, int] = (1, 1, 1)
    seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 7)
    future_steps: int = 14
    model_type: str = "forecasting_sarimax"

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

# ---------------------------------------------------------
# Metrics Tracking Configurations
# ---------------------------------------------------------

@dataclass
class ForecastingMetrics:
    run: str
    mae: float | None = None
    rmse: float | None = None
    elapsed_time_min: float | None = None
    train_loss: List[float] = field(default_factory=list) # Only used for LSTM

@dataclass
class RecommenderMetrics:
    run: str
    epochs: List[int] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    test_accuracy: List[float] = field(default_factory=list)
    elapsed_time_min: float | None = None
    final_accuracy: float | None = None
    
# ---------------------------------------------------------
# Explicit Run Definitions
# ---------------------------------------------------------
# A-Series: Forecasting LSTM Runs (Forecasting + LSTM)
A1 = ForecastingLSTMConfig(run="A1", hidden_size=64, num_layers=1, lookback_window=14, epochs=30)
A2 = ForecastingLSTMConfig(run="A2", hidden_size=128, num_layers=2, lookback_window=21, epochs=50)

# B-Series: Forecasting SARIMAX Runs (Forecasting + SARIMAX)
# B1 = ForecastingSARIMAXConfig(run="B1", order=(1, 0, 1), seasonal_order=(1, 0, 1, 7))
B1 = ForecastingSARIMAXConfig(run="B1", order=(1, 1, 0), seasonal_order=(0, 1, 1, 7))
B2 = ForecastingSARIMAXConfig(run="B2", order=(2, 1, 2), seasonal_order=(1, 1, 1, 7))

# C-Series: Recommender LSTM Runs (Recommender + LSTM)
C1 = RecommenderLSTMConfig(run="C1", embed_dim=64, hidden_dim=128, learning_rate=0.001, epochs=5)
C2 = RecommenderLSTMConfig(run="C2", embed_dim=128, hidden_dim=256, learning_rate=0.005, epochs=10)

def get_run_config(run_id: str):
    return globals().get(run_id)
