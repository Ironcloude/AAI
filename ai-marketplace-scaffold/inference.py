import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import torch.nn.functional as F
from PIL import Image
import os
import sys


def load_model(path):
    print(f"--- Activating model: {os.path.basename(path)} ---", flush=True)
    print("--- loading cpu ---", flush=True)

    model = efficientnet_v2_s(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 2)

    # Load state dict
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    print("--- Model successfully loaded ---", flush=True)
    return model


def predict(model, image_path):
    image = Image.open(image_path).convert("RGB")
    preprocess = EfficientNet_V2_S_Weights.DEFAULT.transforms()
    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, class_idx = torch.max(probabilities, dim=1)

    classes = ["Healthy", "Rotten"]
    return classes[class_idx.item()], float(confidence.item())


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python inference.py <model_path> <image_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]

    model = load_model(model_path)
    label, conf = predict(model, image_path)

    print(f"Prediction: {label}")
    print(f"Confidence: {conf:.4f}")
