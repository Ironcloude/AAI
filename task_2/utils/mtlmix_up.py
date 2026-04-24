import torch
import math
from torchvision.transforms import v2

class _MTLBaseMixUpCutMix(v2.Transform):
    def __init__(self, alpha: float, num_health_classes: int, num_type_classes: int) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self._dist = torch.distributions.Beta(torch.tensor([alpha]), torch.tensor([alpha])) if alpha > 0 else None
        self.num_health_classes = num_health_classes
        self.num_type_classes = num_type_classes

    def forward(self, *inputs):
        # Gracefully unpack whether passed as a single tuple or individual args.
        # Mirrors torchvision v2's `inputs = inputs if len(inputs) > 1 else inputs[0]`
        # pattern but specialised to the fixed (img, y_health, y_type) signature.
        if len(inputs) == 1 and isinstance(inputs[0], (tuple, list)):
            flat_inputs = inputs[0]
        else:
            flat_inputs = inputs
        if len(flat_inputs) != 3:
            raise ValueError(f"MTL transforms expect (image, y_health, y_type), got {len(flat_inputs)} inputs.")

        img, y_health, y_type = flat_inputs

        # 1. Generate parameters (lam or bounding box) based on the image size
        params = self.make_params(img)
    
        # 2. Transform the Image
        img_out = self.transform_image(img, params)

        # 3. Transform Both Labels (using the adjusted lambda)
        lam = params.get("lam_adjusted", params.get("lam"))
        y_health_out = self._mixup_label(y_health, self.num_health_classes, lam)
        y_type_out = self._mixup_label(y_type, self.num_type_classes, lam)

        return img_out, y_health_out, y_type_out

    def _mixup_label(self, label: torch.Tensor, num_classes: int, lam: float) -> torch.Tensor:
        if label.ndim == 1:
            label = torch.nn.functional.one_hot(label, num_classes=num_classes)
        if not label.dtype.is_floating_point:
            label = label.float()
        return label.roll(1, 0).mul_(1.0 - lam).add_(label.mul(lam))

    def make_params(self, img: torch.Tensor) -> dict:
        raise NotImplementedError

    def transform_image(self, img: torch.Tensor, params: dict) -> torch.Tensor:
        raise NotImplementedError
    
class MTLMixUp(_MTLBaseMixUpCutMix):
    def make_params(self, img: torch.Tensor) -> dict:
        # BYPASS: If alpha is 0, return lambda=1.0 (No mixing)
        if self.alpha == 0.0:
            return dict(lam=1.0)
        # Mirrors torchvision.transforms.v2.MixUp.make_params
        return dict(lam=float(self._dist.sample(())))

    def transform_image(self, img: torch.Tensor, params: dict) -> torch.Tensor:
        lam = params["lam"]
        return img.roll(1, 0).mul_(1.0 - lam).add_(img.mul(lam))


class MTLCutMix(_MTLBaseMixUpCutMix):
    def make_params(self, img: torch.Tensor) -> dict:
        # BYPASS: If alpha is 0, create an empty bounding box and lambda=1.0
        if self.alpha == 0.0:
            return dict(box=(0, 0, 0, 0), lam_adjusted=1.0)
        
        lam = float(self._dist.sample(()))
        H, W = img.shape[-2:]

        r_x = torch.randint(W, size=(1,))
        r_y = torch.randint(H, size=(1,))

        r = 0.5 * math.sqrt(1.0 - lam)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))

        lam_adjusted = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))
        return dict(box=(x1, y1, x2, y2), lam_adjusted=lam_adjusted)

    def transform_image(self, img: torch.Tensor, params: dict) -> torch.Tensor:
        x1, y1, x2, y2 = params["box"]
        rolled = img.roll(1, 0)
        output = img.clone()
        output[..., y1:y2, x1:x2] = rolled[..., y1:y2, x1:x2]
        return output