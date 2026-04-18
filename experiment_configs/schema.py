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
    mtl: bool = False
    mtl_primary_weight: float = 0.8
    display_name: str = field(init=False, default="")
    @property
    def weight_string(self) -> str:
        try: 
            return torch_weight_map.get(self.architecture)
        except Exception as e:
            print(f"ERROR: {e}\n"
                  f"Architecture '{self.architecture}' not found in weight map.\n"
                  f"Check torch_weight_map in task_2_config")
        

torch_weight_map = {"efficientnet_v2_s": "EfficientNet_V2_S_Weights.IMAGENET1K_V1"}

