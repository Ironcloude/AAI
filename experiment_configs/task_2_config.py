"""
Define experiments configurations.

Dataclasses used for intellisense.
"""
import sys
sys.path.append(".")
from torchvision.transforms import v2 as T
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

# --- Efficientnet v2 (efficientnet_v2_) experiments ---
# EX-1 - Frozen efficientnet_v2_s baseline
EX1_EFFICIENTNET_FREEZE = Experiment(
    architecture="efficientnet_v2_s",
    training=Training(),
    optimizer= "adamw"
)

# EX-2 - Finetuned efficientnet_v2_s baseline
EX2_EFFICIENTNET_FINETUNE = Experiment(
    architecture="efficientnet_v2_s",
    training=Training(transfer_type="FINETUNE"),
    optimizer= "adamw"
)

# EX-3 - MTL finetuned efficientnet_v2_s
EX3_EFFICIENTNET_FINETUNE_MTL = Experiment(
    architecture="efficientnet_v2_s",
    training=Training(transfer_type="FINETUNE"),
    optimizer= "adamw",
    primary_task_weight=0.8,
)

# --- Swin Transformer (swin_s) experiments ---
# Swin uses a shifted-window attention mechanism instead of convolutions.
# We use a slightly lower learning rate since Transformers are more sensitive to LR.

# EX-4 - Frozen swin_s baseline (transfer learning only)
EX4_SWIN_FREEZE = Experiment(
    architecture="swin_s",
    training=Training(transfer_type="FREEZE", learning_rate=1e-5),
)

# EX-5 - Finetuned swin_s
EX5_SWIN_FINETUNE = Experiment(
    architecture="swin_s",
    training=Training(transfer_type="FINETUNE", learning_rate=1e-5),
)

# EX-6 - MTL finetuned swin_s
EX6_SWIN_FINETUNE_MTL = Experiment(
    architecture="swin_s",
    training=Training(transfer_type="FINETUNE", learning_rate=1e-5),
    primary_task_weight=0.8,
)

# --- MaxViT Hybrid (maxvit_t) experiments ---
# MaxViT combines local convolution blocks with global attention - a hybrid approach.
# Uses same conservative LR as Swin due to attention components.

# EX-7 - Frozen maxvit_t baseline (transfer learning only)
EX7_MAXVIT_FREEZE = Experiment(
    architecture="maxvit_t",
    training=Training(transfer_type="FREEZE", learning_rate=1e-5),
)

# EX-8 - Finetuned maxvit_t
EX8_MAXVIT_FINETUNE = Experiment(
    architecture="maxvit_t",
    training=Training(transfer_type="FINETUNE", learning_rate=1e-5),
)

# EX-9 - MTL finetuned maxvit_t
EX9_MAXVIT_FINETUNE_MTL = Experiment(
    architecture="maxvit_t",
    training=Training(transfer_type="FINETUNE", learning_rate=1e-5),
    primary_task_weight=0.8,
)
# --- Aug experiments ---
# EX-10 - MTL finetuned efficientnet_v2_s aug 7
EX10_EFFICIENTNET_FINETUNE_AUG = Experiment(
    architecture="efficientnet_v2_s",
    training=Training(transfer_type="FINETUNE"),
    augment = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandAugment(num_ops=1, magnitude=7),
            ])
)


# EX-11 - MTL finetuned efficientnet_v2_s aug 10
EX11_EFFICIENTNET_FINETUNE_AUG = Experiment(
    architecture="efficientnet_v2_s",
    training=Training(transfer_type="FINETUNE"),
    augment = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandAugment(num_ops=1, magnitude=10),
            ])
)


# EX-12 - MTL finetuned efficientnet_v2_s aug 13
EX12_EFFICIENTNET_FINETUNE_AUG = Experiment(
    architecture="efficientnet_v2_s",
    training=Training(transfer_type="FINETUNE"),
    augment = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandAugment(num_ops=1, magnitude=13),
            ])
)


assign_display_names(sys.modules[__name__])