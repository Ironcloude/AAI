"""
Define experiments configurations.

Dataclasses used for intellisense.
"""
import sys
sys.path.append(".")
from .schema import Experiment, Scheduler, Training, assign_display_names

# EXPERIMENTS

# EX-1 - Frozen efficientnet_v2_s baseline
EX1_EFFICIENTNET_FREEZE = Experiment(
    architecture="efficientnet_v2_s",
    training=Training(),
    scheduler=Scheduler(),
)

# EX-2 - Finetuned efficientnet_v2_s baseline
EX2_EFFICIENTNET_FINETUNE = Experiment(
    architecture="efficientnet_v2_s",
    training=Training(transfer_type="FINETUNE"),
    scheduler=Scheduler(),
)

# EX-3 - MTL finetuned efficientnet_v2_s
EX3_EFFICIENTNET_FINETUNE_MTL = Experiment(
    architecture="efficientnet_v2_s",
    training=Training(transfer_type="FINETUNE"),
    scheduler=Scheduler(),
    primary_task_weight=0.8,
)

assign_display_names(sys.modules[__name__])
