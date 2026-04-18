"""Multi-task classifier wrapper for produce health and type prediction."""

import torch
import torch.nn as nn
from torchvision.models import get_model

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

        Returns:
            Tuple of (backbone, in_features, dropout_prob).
        """
        if hasattr(base_model, "classifier"):
            head = base_model.classifier

            # EfficientNet / MaxViT: classifier is Sequential([Dropout, Linear])
            if isinstance(head, nn.Sequential):
                print("MTL: Original STL classifier:", base_model.classifier)
                in_features = None
                dropout_prob = 0.0
                for layer in head:
                    if isinstance(layer, nn.Dropout):
                        dropout_prob = layer.p
                    if isinstance(layer, nn.Linear):
                        in_features = layer.in_features
                base_model.classifier = nn.Identity()
                return base_model, in_features, dropout_prob

            # Fallback: plain Linear
            elif isinstance(head, nn.Linear):
                print("MTL: Original STL classifier:", base_model.classifier)
                in_features = head.in_features
                base_model.classifier = nn.Identity()
                return base_model, in_features, 0.0

            else:
                raise Exception(f"Unexpected classifier structure: {type(head)}")

        elif hasattr(base_model, "head"):
            # Swin Transformer: model.head is a single Linear layer
            # e.g. Linear(in_features=768, out_features=1000)
            print("MTL: Original STL head:", base_model.head)
            head = base_model.head
            if isinstance(head, nn.Linear):
                in_features = head.in_features
                base_model.head = nn.Identity()
                return base_model, in_features, 0.0
            else:
                raise Exception(f"Unexpected head structure for Swin: {type(head)}")

        else:
            raise Exception(
                f"Unsupported architecture: {type(base_model).__name__}. "
                "Add its head-stripping logic to _strip_head()."
            )

    def forward(
        self, image_batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (health_logits, type_logits) for the given input batch."""
        features = self.backbone(image_batch)

        # MaxViT's backbone returns a 4D tensor (batch, channels, H, W)
        # We need to flatten it to 2D (batch, features) before the heads
        if features.dim() == 4:
            features = features.mean(dim=[2, 3])  # Global average pooling

        return self.health_head(features), self.type_head(features)

if __name__ == "__main__":
    base = get_model("efficientnet_v2_s", weights=None)
    print("Original classifier:", base.classifier)
