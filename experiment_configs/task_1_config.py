"""
Define experiments configurations for recommendation models.
"""
from dataclasses import dataclass

@dataclass
class Scheduler:
    type: str="StepLR"
    step_size: int=5
    gamma: float=0.1

@dataclass
class Training:
    learning_rate: float=1e-3
    batch_size: int=32
    max_epochs: int=10

@dataclass
class ModelConfig:
    embedding_dim: int=32
    #
    hidden_dim: int=64
    max_seq_len: int=10
    num_heads: int=2
    num_layers: int=2

@dataclass
class SARIMAXConfig:
    order: tuple=(1,1,1)
    seasonal_order:tuple=(1,1,1,7)

@dataclass
class Experiment:
    name: str
    architecture: str
    training: Training
    scheduler: Scheduler
    model: ModelConfig=None
    sarimax: SARIMAXConfig=None

#EXPERIMENTS

ncf_baseline=Experiment(
    name="NCF_BASELINE",
    architecture="ncf",
    training=Training(learning_rate=1e-3),
    scheduler=Scheduler(),
    model=ModelConfig(embedding_dim=32)
)

lstm_baseline = Experiment(
    name="LSTM_BASELINE",
    architecture="lstm",
    training=Training(learning_rate=1e-3),
    scheduler=Scheduler(),
    model=ModelConfig(
        embedding_dim=32,
        hidden_dim=64,
        max_seq_len=10
    )
)

sasrec_baseline=Experiment(
    name="SASREC_BASELINE",
    architecture="sasrec",
    training=Training(learning_rate=1e-4), #less for transformer models
    scheduler=Scheduler(),
    model=ModelConfig(
        embedding_dim=32,
        max_seq_len=10,
        num_heads=2,
        num_layers=2
    )
)

sarimax_baseline=Experiment(
    name="SARIMAX_BASELINE",
    architecture="sarimax",
    training=Training(learning_rate=1e-3),
    scheduler=Scheduler(),
    sarimax=SARIMAXConfig(
        order=(1,1,1),
        seasonal_order=(1,1,1,7)
    )
)