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
    learning_rate: float = 1e-4
    momentum: float = 0.9
    batch_size: int = 32
    max_epochs: int = 10
    augment: bool = False
    optimizer: str = "Adam"       # "Adam" or "SGD"
    weight_decay: float = 1e-4 
    class_weights: bool = False   # weight loss by inverse class frequency


@dataclass
class Experiment:
    display_name: str
    architecture: str
    weights: str
    training: Training
    scheduler: Scheduler

# EXPERIMENTS

# EX-1 - Frozen efficientnet_v2_s baseline
efficientnet_freeze_baseline = Experiment(
    display_name="EX1_efficientnet_freeze_baseline",
    architecture="efficientnet_v2_s",
    weights="EfficientNet_V2_S_Weights.IMAGENET1K_V1",
    training=Training(),
    scheduler=Scheduler(),
)

# EX-2 - Finetuned efficientnet_v2_s baseline
efficientnet_finetune_baseline = Experiment(
    display_name="EX2_efficientnet_finetune_baseline",
    architecture="efficientnet_v2_s",
    weights="EfficientNet_V2_S_Weights.IMAGENET1K_V1",
    training=Training(transfer_type="FINETUNE"),
    scheduler=Scheduler(),
)

# EX-3 - Finetuned efficientnet_v2_s (augmentation)
efficientnet_finetune_augment_cw = Experiment(
    display_name="EX3_efficientnet_finetune_augment",
    architecture="efficientnet_v2_s",
    weights="EfficientNet_V2_S_Weights.IMAGENET1K_V1",
    training=Training(transfer_type="FINETUNE", augment=True),
    scheduler=Scheduler(),
)

# EX-4 - Finetuned efficientnet_v2_s (Adam + class weights + augmentation)
efficientnet_finetune_adam_cw = Experiment(
    display_name="EX4_efficientnet_finetune_adam_cw",
    architecture="efficientnet_v2_s",
    weights="EfficientNet_V2_S_Weights.IMAGENET1K_V1",
    training=Training(transfer_type="FINETUNE", augment=True, optimizer="Adam", class_weights=True),
    scheduler=Scheduler(),
)

# EX-5 - Finetuned efficientnet_v2_s (Adam + class weights, no augmentation)
efficientnet_finetune_adam_cw_noaug = Experiment(
    display_name="EX5_efficientnet_finetune_adam_cw_noaug",
    architecture="efficientnet_v2_s",
    weights="EfficientNet_V2_S_Weights.IMAGENET1K_V1",
    training=Training(transfer_type="FINETUNE", optimizer="Adam", class_weights=True),
    scheduler=Scheduler(),
)