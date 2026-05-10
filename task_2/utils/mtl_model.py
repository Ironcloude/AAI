"""Multi-task classifier wrapper for produce health and type prediction."""

import sys
import torch
import torch.nn as nn
from torchvision.models import get_model, get_weight

class MultiTaskClassifier(nn.Module):
    """Architecture-agnostic multi-task wrapper.

    Takes any torchvision model, strips its classification head and
    attaches two task-specific heads (health and produce type).
    """

    def __init__(self, base_model: nn.Module, num_produce_classes: int, 
                 num_health_classes: int = 2,) -> None:
        """Initialise the wrapper around a torchvision backbone.

        Args:
            base_model: Torchvision model to wrap (e.g. EfficientNet).
            num_produce_classes: Number of distinct produce classes.
            num_health_classes: Output size of the health head.
        """
        super().__init__()

        # Detect and remove the existing classifier head, extracting its input
        # dimensionality and dropout prob
        self.backbone, in_features, dropout_p = self._strip_head(base_model)

        # Head #1 - Health Classification (Healthy [0] vs. Rotten [1])
        self.health_head = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_health_classes)
        )
        # Head #2 - Produce type Classification (e.g., Apple, Banana, ..., etc.)
        self.type_head = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_produce_classes)
        )

    @staticmethod
    def _strip_head(base_model: nn.Module) -> tuple[nn.Module, int, float]:
        """Remove the final classification layer from a torchvision model.

        Torchvision models use different attribute names for their classifier:
          - EfficientNet: model.classifier = Sequential(Dropout, Linear)
          - Swin:         model.head = Linear
          - MaxViT:       model.classifier = Sequential(..., Linear)

        Replace the original head/neck with Identity to preserve pretrained features.

        Returns:
            Tuple of (backbone, in_features, dropout_prob).
        """

        # Locate head attribute (torchvision uses .classifier or .head)
        if hasattr(base_model, "classifier"): # Efficientnet / MaxViT
            head_attr = "classifier"
        elif hasattr(base_model, "head"): # Swin
            head_attr = "head"
        else:
            raise Exception(f"Unsupported architecture: {type(base_model).__name__}")

        classification_head = getattr(base_model, head_attr)
        print(f"[MTL] Detected head attr: '.{head_attr}'  arch={type(base_model).__name__}")
        print(f"[MTL] Original head:\n{classification_head}")

        # EfficientNet / MaxViT: classifier is Sequential([Dropout, Linear])
        # Strip only the FINAL layer (keep any intermediate Dropout, etc.)
        dropout_prob = 0.0
        if isinstance(classification_head, nn.Sequential):
            in_features = classification_head[-1].in_features
            # Dropout prob: look for Dropout sitting just before the final Linear
            dropout_prob = next(
                (layer.p for layer in classification_head[:-1]if isinstance(layer, nn.Dropout)),
                0.0,
            )
            
            classification_head[-1] = nn.Identity()  

        # Swin transformer: plain Linear; swap everything
        elif isinstance(classification_head, nn.Linear):
            in_features = classification_head.in_features
            base_model.head = nn.Identity()  
        
        else:
            raise Exception(
                f"Unsupported architecture: {type(base_model).__name__}. "
                "Add its head-stripping logic to _strip_head()."
            )
        
        # Validate output shape with a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feats = base_model(dummy)
            print(f"[MTL] Backbone+neck output shape: {tuple(feats.shape)}  "
                f"(expected (1, {in_features}))")
            assert feats.shape == (1, in_features), (
                f"Expected backbone output shape (1, {in_features}), got {feats.shape}"
            )
        return base_model, in_features, dropout_prob

    def forward(
        self, image_batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (health_logits, type_logits) for the given input batch."""
        features = self.backbone(image_batch)
        return self.health_head(features), self.type_head(features)

if __name__ == "__main__":
    for exp in [experiments.EX3_EFFICIENTNET_FINETUNE_MTL, experiments.EX6_SWIN_FINETUNE_MTL,
                experiments.EX9_MAXVIT_FINETUNE_MTL]:
        print(f"\nInstantiating experiment: {exp.display_name}")
        pretrained_weights = get_weight(exp.weight_string)
        PRETRAINED_MODEL = get_model(exp.architecture, weights=pretrained_weights)
        model = MultiTaskClassifier(PRETRAINED_MODEL, num_produce_classes=12312)
