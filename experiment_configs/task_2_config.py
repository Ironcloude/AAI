"""
Define experiments configurations.

Dataclasses used for intellisense.
"""
from dataclasses import dataclass

@dataclass
class Scheduler:
    type: str = "StepLR"
    step_size: int = 5
    gamma: float = 0.1

@dataclass
class Training:
    transfer_type: str = "FREEZE"
    learning_rate: float = 1e-3
    momentum: float = 0.9
    batch_size: int = 32
    max_epochs: int = 10

@dataclass
class Experiment:
    architecture: str
    weights: str
    training: Training
    scheduler: Scheduler

# EXPERIMENTS

# EX-1 - Frozen efficientnet_v2_s baseline
efficientnet_freeze_baseline = Experiment(
    architecture="efficientnet_v2_s",
    weights="EfficientNet_V2_S_Weights.IMAGENET1K_V1",
    training=Training(),
    scheduler=Scheduler(),
)

# EX-2 - Finetuend efficientnet_v2_s baseline
efficientnet_finetune_baseline = Experiment(
    architecture="efficientnet_v2_s",
    weights="EfficientNet_V2_S_Weights.IMAGENET1K_V1",
    training=Training(transfer_type="FINETUNE", learning_rate=1e-4),
    scheduler=Scheduler(),
)
