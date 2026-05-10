import torch
# 1. Reshape Swin tokens back into a 7x7 2D feature map
def swin_reshape_transform(tensor, height=7, width=7):
    """
    Dynamically reshapes the Swin outputs depending on the backend (timm vs torchvision)
    Grad-CAM demands: [Batch, Channels, Height, Width]
    """
    # 1. PyTorch/Torchvision format: [Batch, Height, Width, Channels]
    if len(tensor.shape) == 4:
        # Move the Channels (dim 3) to the front (dim 1)
        # ->
        return tensor.permute(0, 3, 1, 2)
        
    # 2. HuggingFace/timm format: [Batch, Sequence (49), Channels]
    elif len(tensor.shape) == 3:
        result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
        result = result.permute(0, 3, 1, 2)
        return result
        
    return tensor
# 2. Wrapper to isolate one task for the gradient calculation
class TaskWrapper(torch.nn.Module):
    def __init__(self, base_model, target_task='health'):
        super().__init__()
        self.base_model = base_model
        self.target_task = target_task

    def forward(self, x):
        health, type_ = self.base_model(x)
        return health if self.target_task == 'health' else type_

