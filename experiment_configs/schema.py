from dataclasses import dataclass, field


def assign_display_names(module) -> None:
    """Generates 'display name' attribute 
        Based on Experiment name
    """
    for name, obj in vars(module).items():
        if isinstance(obj, Experiment):
            obj.display_name = name

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
    optimizer: str = "adamw"
    primary_task_weight: float = 1.0
    display_name: str = field(init=False, default="")
    @property
    def is_mtl(self) -> bool:
        """True iff type head receives non-zero loss weighting."""
        return self.primary_task_weight < 1.0

    @property
    def weight_string(self) -> str:
        try: 
            return torch_weight_map.get(self.architecture)
        except Exception as e:
            print(f"ERROR: {e}\n"
                  f"Architecture '{self.architecture}' not found in weight map.\n"
                  f"Check torch_weight_map in task_2_config")
        

# Maps architecture name -> torchvision pretrained weights string
# Added swin_s (Transformer) and maxvit_t (Hybrid) to support new experiments
torch_weight_map = {
    "efficientnet_v2_s": "EfficientNet_V2_S_Weights.IMAGENET1K_V1",
    "swin_s":            "Swin_S_Weights.IMAGENET1K_V1",
    "maxvit_t":          "MaxVit_T_Weights.IMAGENET1K_V1",
}