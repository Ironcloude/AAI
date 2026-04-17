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

@dataclass
class Experiment:
    display_name: str
    architecture: str
    training: Training
    scheduler: Scheduler
    mtl: bool = False
    mtl_primary_weight: float = 0.8
    @property
    def weight_string(self) -> str:
        try: 
            return torch_weight_map.get(self.architecture)
        except Exception as e:
            print(f"ERROR: {e}\n"
                  f"Architecture '{self.architecture}' not found in weight map.\n"
                  f"Check torch_weight_map in task_2_config")
        

torch_weight_map = {"efficientnet_v2_s": "EfficientNet_V2_S_Weights.IMAGENET1K_V1"}

def get_var_name(obj, module):
    """Get experiment var name for display"""
    for name in dir(module):
        if getattr(module, name) is obj:
            return name
    return "Unknown_Experiment"


# EXPERIMENTS

# EX-1 - Frozen efficientnet_v2_s baseline
EX1_EFFICIENTNET_FREEZE = Experiment(
    display_name="EX1_efficientnet_freeze_baseline",
    architecture="efficientnet_v2_s",
    training=Training(),
    scheduler=Scheduler(),
)

# EX-2 - Finetuned efficientnet_v2_s baseline
EX2_EFFICIENTNET_FINETUNE = Experiment(
    display_name="EX2_efficientnet_finetune_baseline",
    architecture="efficientnet_v2_s",
    training=Training(transfer_type="FINETUNE"),
    scheduler=Scheduler(),
)

# EX-3 - MTL finetuned efficientnet_v2_s
EX3_EFFICIENTNET_MTL_FINETUNE = Experiment(
    display_name="EX3_efficientnet_finetune_mtl",
    architecture="efficientnet_v2_s",
    training=Training(transfer_type="FINETUNE"),
    scheduler=Scheduler(),

)

