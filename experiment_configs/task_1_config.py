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
    batch_size: int=64
    max_epochs: int=15

@dataclass
class ModelConfig:
    embedding_dim: int=32
    hidden_dim: int=64
    max_seq_len: int=10
    num_heads: int=2
    num_layers: int=2
    dropout: float=0.1

# @dataclass
# class SARIMAXConfig:
#     order: tuple=(1,1,1)
#     seasonal_order:tuple=(1,1,1,7)

@dataclass
class Experiment:
    name: str
    display_name: str
    architecture: str
    training: Training
    scheduler: Scheduler
    model: ModelConfig=None
#    sarimax: SARIMAXConfig=None

#EXPERIMENTS

ncf_baseline=Experiment(
    name="NCF_BASELINE",
    display_name="NCF_Base",
    architecture="ncf",
    training=Training(learning_rate=1e-3),
    scheduler=Scheduler(),
    model=ModelConfig(embedding_dim=32)
)

ncf_deep=Experiment(
    name="NCF_DEEP",
    display_name="NCF_Deep_Layers",
    architecture="ncf",
    training=Training(learning_rate=5e-4,max_epochs=20),
    scheduler=Scheduler(step_size=7),
    model=ModelConfig(embedding_dim=64)
)

lstm_baseline = Experiment(
    name="LSTM_BASELINE",
    display_name="LSTM_Base",
    architecture="lstm",
    training=Training(learning_rate=1e-3),
    scheduler=Scheduler(),
    model=ModelConfig(
        embedding_dim=32,
        hidden_dim=64,
        max_seq_len=10
    )
)

lstm_long_context = Experiment(
    name="LSTM_LONG_CONTEXT",
    display_name="LSTM_Seq20",
    architecture="lstm",
    training=Training(learning_rate=1e-3,batch_size=32),
    scheduler=Scheduler(),
    model=ModelConfig(
        embedding_dim=32,
        hidden_dim=128,
        max_seq_len=20
    )
)

sasrec_baseline=Experiment(
    name="SASREC_BASELINE",
    display_name="SASRec_Base",
    architecture="sasrec",
    training=Training(learning_rate=1e-4), #less for transformer models
    scheduler=Scheduler(step_size=10),
    model=ModelConfig(
        embedding_dim=32,
        max_seq_len=10,
        num_heads=4,
        num_layers=2
    )
)

sasrec_heavy=Experiment(
    name="SASREC_HEAVY",
    display_name="SASRec_Deep_Attn",
    architecture="sasrec",
    training=Training(
        learning_rate=1e-4,
        max_epochs=20
    ),
    scheduler=Scheduler(),
    model=ModelConfig(
        embedding_dim=64,
        max_seq_len=15,
        num_heads=8,
        num_layers=4,
    )
)