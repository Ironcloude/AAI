"""
Define experiments configurations.

Dataclasses used for intellisense.
"""
import sys
sys.path.append(".")
from .schema import Experiment, Scheduler, Training, assign_display_names

# EXPERIMENTS

# --- Efficientnet v2 (efficientnet_v2_) experiments ---
# EX-1 - Frozen efficientnet_v2_s baseline
EX1_EFFICIENTNET_FREEZE = Experiment(
    architecture="efficientnet_v2_s",
    training=Training(),
    scheduler=Scheduler(),
    optimizer= "sgd"
)

# EX-2 - Finetuned efficientnet_v2_s baseline
EX2_EFFICIENTNET_FINETUNE = Experiment(
    architecture="efficientnet_v2_s",
    training=Training(transfer_type="FINETUNE"),
    scheduler=Scheduler(),
    optimizer= "sgd"

)


# EX-3 - MTL finetuned efficientnet_v2_s
EX3_EFFICIENTNET_FINETUNE_MTL = Experiment(
    architecture="efficientnet_v2_s",
    training=Training(transfer_type="FINETUNE"),
    scheduler=Scheduler(),
    optimizer= "sgd",
    primary_task_weight=0.8,
)

# --- Swin Transformer (swin_s) experiments ---
# Swin uses a shifted-window attention mechanism instead of convolutions.
# We use a slightly lower learning rate since Transformers are more sensitive to LR.

# EX-4 - Frozen swin_s baseline (transfer learning only)
EX4_SWIN_FREEZE = Experiment(
    architecture="swin_s",
    training=Training(transfer_type="FREEZE", learning_rate=1e-5),
    scheduler=Scheduler(),
)

# EX-5 - Finetuned swin_s
EX5_SWIN_FINETUNE = Experiment(
    architecture="swin_s",
    training=Training(transfer_type="FINETUNE", learning_rate=1e-5),
    scheduler=Scheduler(),
)

# EX-6 - MTL finetuned swin_s
EX6_SWIN_FINETUNE_MTL = Experiment(
    architecture="swin_s",
    training=Training(transfer_type="FINETUNE", learning_rate=1e-5),
    scheduler=Scheduler(),
)

# --- MaxViT Hybrid (maxvit_t) experiments ---
# MaxViT combines local convolution blocks with global attention - a hybrid approach.
# Uses same conservative LR as Swin due to attention components.

# EX-7 - Frozen maxvit_t baseline (transfer learning only)
EX7_MAXVIT_FREEZE = Experiment(
    architecture="maxvit_t",
    training=Training(transfer_type="FREEZE", learning_rate=1e-5),
    scheduler=Scheduler(),
)

# EX-8 - Finetuned maxvit_t
EX8_MAXVIT_FINETUNE = Experiment(
    architecture="maxvit_t",
    training=Training(transfer_type="FINETUNE", learning_rate=1e-5),
    scheduler=Scheduler(),
)

# EX-9 - MTL finetuned maxvit_t
EX9_MAXVIT_FINETUNE_MTL = Experiment(
    architecture="maxvit_t",
    training=Training(transfer_type="FINETUNE", learning_rate=1e-5),
    scheduler=Scheduler(),
)

assign_display_names(sys.modules[__name__])
