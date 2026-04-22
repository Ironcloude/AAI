import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from PIL import Image
import numpy as np
import os
import sys
import cv2
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from safetensors.torch import load_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
class MultiTaskClassifier(nn.Module):
    def __init__(self, base_model, num_produce_classes, num_health_classes=2):
        super().__init__()
        self.backbone = base_model
        # Strip the head like in the original code
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.health_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_features, num_health_classes)
        )
        self.type_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_features, num_produce_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.health_head(features), self.type_head(features)


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx, is_mtl=False):
        self.model.zero_grad()
        output = self.model(input_tensor)

        # Handle MTL vs STL output
        if is_mtl:
            health_logits, _ = output
            logits = health_logits
        else:
            logits = output

        loss = logits[0, class_idx]
        loss.backward()

        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam_min = np.min(cam)
        cam_max = np.max(cam)
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cam


class GuidedBackprop:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                self.hooks.append(
                    module.register_forward_hook(self.forward_hook))
                self.hooks.append(
                    module.register_full_backward_hook(self.backward_hook))

    def forward_hook(self, module, input, output):
        self.forward_relu_outputs = output

    def backward_hook(self, module, grad_in, grad_out):
        grad = grad_out[0]
        positive_grad = torch.clamp(grad, min=0.0)
        positive_output = torch.clamp(self.forward_relu_outputs, min=0.0)
        return (positive_grad * (positive_output > 0).float(),)

    def generate(self, input_tensor, class_idx, is_mtl=False):
        input_tensor.requires_grad = True
        self.model.zero_grad()
        output = self.model(input_tensor)

        if is_mtl:
            health_logits, _ = output
            logits = health_logits
        else:
            logits = output

        loss = logits[0, class_idx]
        loss.backward()

        return input_tensor.grad.detach().cpu().numpy()[0]


def load_model(path):
    print(f"--- Activating model: {os.path.basename(path)} ---")
    if path.endswith(".safetensors"):
        if not HAS_SAFETENSORS:
            raise ImportError("safetensors library not found. Please install it via 'pip install safetensors'.")
        state_dict = load_file(path, device="cpu")
    else:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)

    # Detect architecture: Multi-Task (MTL) vs Single-Task (STL)
    is_mtl = 'type_head.1.weight' in state_dict
    base_model = efficientnet_v2_s(weights=None)

    if is_mtl:
        print("Detected Multi-Task (MTL) architecture")
        num_produce_classes = state_dict['type_head.1.weight'].shape[0]
        model = MultiTaskClassifier(
            base_model, num_produce_classes=num_produce_classes)
    else:
        print("Detected Single-Task (STL) architecture")
        num_features = base_model.classifier[1].in_features
        base_model.classifier[1] = nn.Linear(num_features, 2)
        model = base_model

    model.load_state_dict(state_dict)
    model.eval()
    return model, is_mtl


def create_visualizations(image_np, cam, guided_grads, label, conf, title_suffix):
    # Prepare image for plotly (H, W, C)
    img_rgb = image_np

    # Guided Grad-CAM
    cam_3d = np.expand_dims(cam, axis=2)
    guided_cam = guided_grads.transpose(1, 2, 0) * cam_3d
    guided_cam_min = np.min(guided_cam)
    guided_cam_max = np.max(guided_cam)
    if guided_cam_max > guided_cam_min:
        guided_cam = (guided_cam - guided_cam_min) / (guided_cam_max - guided_cam_min + 1e-8)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Original Image",
            f"Grad-CAM (Attention on {label})",
            "Guided Backprop",
            "Guided Grad-CAM"
        )
    )

    # 1. Original
    fig.add_trace(go.Image(z=(img_rgb * 255).astype(np.uint8)), row=1, col=1)

    # 2. Grad-CAM
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    overlaid = (heatmap * 0.4 + img_rgb * 0.6)
    fig.add_trace(go.Image(z=(overlaid * 255).astype(np.uint8)), row=1, col=2)

    # 3. Guided Backprop
    gb = guided_grads.transpose(1, 2, 0)
    gb_min = np.min(gb)
    gb_max = np.max(gb)
    if gb_max > gb_min:
        gb_norm = (gb - gb_min) / (gb_max - gb_min + 1e-8)
    else:
        gb_norm = gb
    fig.add_trace(go.Image(z=(gb_norm * 255).astype(np.uint8)), row=2, col=1)

    # 4. Guided Grad-CAM
    fig.add_trace(go.Image(z=(guided_cam * 255).astype(np.uint8)), row=2, col=2)

    fig.update_layout(
        title=f"XAI Analysis: {label} ({conf:.2%} confidence) | {title_suffix}",
        height=900, width=1000
    )

    output_html = "xai_result.html"
    fig.write_html(output_html)
    print(f"Visualization saved to {output_html}")
    fig.show()


def main():
    if len(sys.argv) < 3:
        print("Usage: python inference_xai.py <model_path> <image_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]

    model, is_mtl = load_model(model_path)

    # Preprocessing
    image = Image.open(image_path).convert("RGB")
    preprocess = EfficientNet_V2_S_Weights.DEFAULT.transforms()
    input_tensor = preprocess(image).unsqueeze(0)

    # Use original high-res image for plotting (Avoid pixelation)
    img_display = np.array(image) / 255.0
    h_orig, w_orig = img_display.shape[:2]

    # Predictions
    with torch.no_grad():
        output = model(input_tensor)
        if is_mtl:
            health_logits, type_logits = output
            health_probs = F.softmax(health_logits, dim=1)
            type_probs = F.softmax(type_logits, dim=1)
            h_conf, h_idx = torch.max(health_probs, dim=1)
            t_conf, t_idx = torch.max(type_probs, dim=1)
            title_suffix = f"TypeIdx_{t_idx.item()}"
        else:
            probs = F.softmax(output, dim=1)
            h_conf, h_idx = torch.max(probs, dim=1)
            title_suffix = "STL Mode"

    health_labels = ["Healthy", "Rotten"]
    print(f"Prediction: {health_labels[h_idx]} ({h_conf.item():.4f})")
    if is_mtl:
        print(f"Type Index: {t_idx.item()} ({t_conf.item():.4f})")

    # XAI Generation
    if is_mtl:
        target_layer = model.backbone.features[-1]
    else:
        target_layer = model.features[-1]

    gcam = GradCAM(model, target_layer)
    gbp = GuidedBackprop(model)

    cam = gcam.generate(input_tensor, h_idx, is_mtl=is_mtl)
    guided_grads = gbp.generate(input_tensor, h_idx, is_mtl=is_mtl)

    # Upscale XAI maps to match high-res display image
    cam_high_res = cv2.resize(cam, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
    
    guided_grads_res = guided_grads.transpose(1, 2, 0)
    guided_grads_high_res = cv2.resize(guided_grads_res, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
    # Convert back to (C, H, W) for visualization function consistency if needed, 
    # but we'll update create_visualizations to handle (H, W, C) directly for grads.
    guided_grads_high_res = guided_grads_high_res.transpose(2, 0, 1)

    create_visualizations(img_display, cam_high_res, guided_grads_high_res,
                          health_labels[h_idx], h_conf.item(), title_suffix)


if __name__ == "__main__":
    main()
