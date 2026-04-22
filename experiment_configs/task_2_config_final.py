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
# Efficientnet v2 (CNN)
# EX-1 - Finetuned efficientnet_v2_s baseline
# 21.4m Paramaters, 8.37 GLOPs
EX1_EFFICIENTNET_FINETUNE = Experiment(
    architecture="efficientnet_v2_s",
    training=Training(transfer_type="FINETUNE", learning_rate=1e-4), 
)
# Swin Transformer (Transformer)
# Swin uses a shifted-window attention mechanism instead of convolutions.
# Slightly lower learning rate since Transformers are more sensitive to LR.
# 49.6M Parameters, 8.74 GLOPs
EX2_SWIN_FINETUNE = Experiment(
    architecture="swin_s",
    training=Training(transfer_type="FINETUNE", learning_rate=1e-5), # Defined by Swin paper
)
# MaxViT (Hybrid) combines local convolution blocks with global attention - a hybrid approach.
# Uses similalry conservative LR as Swin due to attention components.
# 30.9m Parameters, 5.56 GLOPs
EX3_MAXVIT_FINETUNE = Experiment(
    architecture="maxvit_t",
    training=Training(transfer_type="FINETUNE", learning_rate=5e-5), # Defined by maxvit paper
)

# Comparable "tiny" variants for fair comparison to maxvit.
# EX-1T - effecientnet_b4, 4.39 GFLOPS
EX1T_EFFICIENTNET_FINETUNE = Experiment(
    architecture="efficientnet_b4",
    training=Training(transfer_type="FINETUNE"),
    batch_size = 8,
    acc_steps =  4
)

# EX-2T - swin_t,  4.49 GLOPS
EX2T_SWIN_FINETUNE = Experiment(
    architecture="swin_t",
    training=Training(transfer_type="FINETUNE", learning_rate=1e-5), 
)

# MTL EXPERIMENTS
EX4a_EFFICIENTNET_FINETUNE_MTL = Experiment(
    architecture="efficientnet_v2_s",
    training=Training(transfer_type="FINETUNE", learning_rate=1e-4), 
    primary_task_weight=0.9
)
EX4b_EFFICIENTNET_FINETUNE_MTL = Experiment(
    architecture="efficientnet_v2_s",
    training=Training(transfer_type="FINETUNE", learning_rate=1e-4), 
    primary_task_weight=0.75
)
EX4c_EFFICIENTNET_FINETUNE_MTL = Experiment(
    architecture="efficientnet_v2_s",
    training=Training(transfer_type="FINETUNE", learning_rate=1e-4),
    primary_task_weight=0.5 
)

EX5a_MAXVIT_FINETUNE_MTL = Experiment(
    architecture="maxvit_t",
    training=Training(transfer_type="FINETUNE", learning_rate=5e-5),
    primary_task_weight=0.9
)

EX5b_MAXVIT_FINETUNE_MTL = Experiment(
    architecture="maxvit_t",
    training=Training(transfer_type="FINETUNE", learning_rate=5e-5),
    primary_task_weight=0.75
)

EX5c_MAXVIT_FINETUNE_MTL = Experiment(
    architecture="maxvit_t",
    training=Training(transfer_type="FINETUNE", learning_rate=5e-5),
    primary_task_weight=0.5
)

EX6a_SWIN_FINETUNE_MTL = Experiment(
    architecture="swin_s",
    training=Training(transfer_type="FINETUNE", learning_rate=1e-5),
    primary_task_weight=0.9
)

EX6b_SWIN_FINETUNE_MTL = Experiment(
    architecture="swin_s",
    training=Training(transfer_type="FINETUNE", learning_rate=1e-5),
    primary_task_weight=0.75
)

EX6c_SWIN_FINETUNE_MTL = Experiment(
    architecture="swin_s",
    training=Training(transfer_type="FINETUNE", learning_rate=1e-5),
    primary_task_weight=0.5
)

# Ablations on winning architecture (Swin-T) 
EX7a_SWIN_MTL_FREEZE = Experiment(
    architecture="swin_s",
    training=Training(transfer_type="FREEZE", learning_rate=1e-4),
    primary_task_weight=0.9
)
EX7b_SWIN_MTL_SCRATCH = Experiment(
    architecture="swin_s",
    training=Training(transfer_type="FINETUNE", learning_rate=1e-5),
    pretrained=False,
    primary_task_weight=0.9
)
EX7c_SWIN_MTL_UNWEIGHTED = Experiment(
    architecture="swin_s",
    training=Training(transfer_type="FINETUNE", learning_rate=1e-5),
    class_weighted=False,
    primary_task_weight=0.9
)

assign_display_names(sys.modules[__name__])