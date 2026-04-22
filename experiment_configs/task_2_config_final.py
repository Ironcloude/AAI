"""
Define experiments configurations.

Dataclasses used for intellisense.
"""
import sys
sys.path.append(".")
from .schema import Experiment, Training, assign_display_names

# EXPERIMENTS
# Goal:
# ID val is over-saturated; selection is based on: OOD validation accuracy (primary), OOD AUC-ROC (Secondary), OOD ECE (Tertiary)
# 1. Identify best dataset variation using EfficientNet STL baseline (deduplication, pre-augmentation)
# 2. Identify best STL architecture at fixed optimizer (AdamW): EfficientNet vs Swin vs MaxViT
# 3. On winning arch, MTL vs STL with unified stopping criterion (primary-task loss)
# 4. On winning arch, architectural ablations:
#    a. freeze / partial-freeze / finetune
#    b. pretrained / random-init
#    c. class-weighted / unweighted loss
# 5. Augmentation ablation on best config from (4)
# 6. Post-hoc: temperature scaling on val > OOD ECE check

# 
# --- Efficientnet v2 (efficientnet_v2_) experiments ---
# EX-1 - Finetuned efficientnet_v2_s baseline
# 21.4m Paramaters, 8.37 GLOPs
EX1_EFFICIENTNET_FINETUNE = Experiment(
    architecture="efficientnet_v2_s",
    training=Training(transfer_type="FINETUNE", learning_rate=1e-4), 
)
# --- Swin Transformer (swin_s) experiments ---
# Swin uses a shifted-window attention mechanism instead of convolutions.
# Slightly lower learning rate since Transformers are more sensitive to LR.
# 49.6M Parameters, 8.74 GLOPs
EX2_SWIN_FINETUNE = Experiment(
    architecture="swin_s",
    training=Training(transfer_type="FINETUNE", learning_rate=1e-5), # Defined by Swin paper
)
# # --- MaxViT Hybrid (maxvit_t) experiments ---
# MaxViT combines local convolution blocks with global attention - a hybrid approach.
# Uses similalry conservative LR as Swin due to attention components.
# 30.9m Parameters, 5.56 GLOPs
EX3_MAXVIT_FINETUNE = Experiment(
    architecture="maxvit_t",
    training=Training(transfer_type="FINETUNE", learning_rate=5e-5), # Defined by maxvit paper
)

# Comparable GLOP comparsion 
# ENET_2 => ENET_B4     4.39
# Swin_s => Swin_v2_t   4.49
# MaxVit_t => Same      5.57    
# EX-1T - Finetuned effecientnet b4 baseline
EX1T_EFFICIENTNET_FINETUNE = Experiment(
    architecture="efficientnet_b4",
    training=Training(transfer_type="FINETUNE"),
    batch_size = 8,
    acc_steps =  4
)

# EX-2T - Finetuned swin_t
EX2T_SWIN_FINETUNE = Experiment(
    architecture="swin_t",
    training=Training(transfer_type="FINETUNE", learning_rate=1e-5), 
)

# EX- - Frozen efficientnet_v2_s baseline
# )
# EX_EFFICIENTNET_FREEZE = Experiment(
#     architecture="efficientnet_v2_s",
#     training=Training(),
#     optimizer= "adamw"
# )

# # EX- - MTL finetuned efficientnet_v2_s
# EX_EFFICIENTNET_FINETUNE_MTL = Experiment(
#     architecture="efficientnet_v2_s",
#     training=Training(transfer_type="FINETUNE"),
#     optimizer= "adamw",
#     primary_task_weight=0.8,
# )


# # EX- Frozen swin_s baseline (transfer learning only)
# EX_SWIN_FREEZE = Experiment(
#     architecture="swin_s",
#     training=Training(transfer_type="FREEZE", learning_rate=1e-5),
# )

# # EX- Finetuned swin_s
# EX_SWIN_FINETUNE = Experiment(
#     architecture="swin_s",
#     training=Training(transfer_type="FINETUNE", learning_rate=1e-5),
# )

# # EX- MTL finetuned swin_s
# EX_SWIN_FINETUNE_MTL = Experiment(
#     architecture="swin_s",
#     training=Training(transfer_type="FINETUNE", learning_rate=1e-5),
#     primary_task_weight=0.8,
# )


# # EX- Frozen maxvit_t baseline (transfer learning only)
# EX_MAXVIT_FREEZE = Experiment(
#     architecture="maxvit_t",
#     training=Training(transfer_type="FREEZE", learning_rate=1e-5),
# )


# # EX- MTL finetuned maxvit_t
# EX_MAXVIT_FINETUNE_MTL = Experiment(
#     architecture="maxvit_t",
#     training=Training(transfer_type="FINETUNE", learning_rate=1e-5),
#     primary_task_weight=0.8,
# )

 
# # EX - MTL finetuned efficientnet_v2_s aug 7
# EX_EFFICIENTNET_FINETUNE_AUG = Experiment(
#     architecture="efficientnet_v2_s",
#     training=Training(transfer_type="FINETUNE"),
#     optimizer= "adamw",
#     aug_magnitude = 7
# )

# # EX - MTL finetuned efficientnet_v2_s aug 10
# EX_EFFICIENTNET_FINETUNE_AUG = Experiment(
#     architecture="efficientnet_v2_s",
#     training=Training(transfer_type="FINETUNE"),
#     optimizer= "adamw",
#     aug_magnitude = 10
# )

# # EX - MTL finetuned efficientnet_v2_s aug 13
# EX_EFFICIENTNET_FINETUNE_AUG = Experiment(
#     architecture="efficientnet_v2_s",
#     training=Training(transfer_type="FINETUNE"),
#     optimizer= "adamw",
#     aug_magnitude = 13
# )

assign_display_names(sys.modules[__name__])