import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pillow_heif
import numpy as np
import xgboost as xgb
import argparse
import pandas as pd

# Register HEIF opener
pillow_heif.register_heif_opener()

# ---------------------------
# Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------
# Dataset
# ---------------------------
class InferenceImageDataset(Dataset):
    """Dataset for inference (no labels required)"""
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, img_path

# ---------------------------
# Transforms
# ---------------------------
val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ---------------------------
# Projection Head
# ---------------------------
import torch.nn as nn
class ProjectionHead(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------
# SSL Backbone
# ---------------------------
def get_ssl_backbone(pretrained=False):
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    return backbone.to(device)

# ---------------------------
# Feature extraction
# ---------------------------
def extract_embeddings(image_paths, backbone, transform, batch_size=8):
    dataset = InferenceImageDataset(image_paths, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    backbone.eval()
    feats = []
    img_paths = []

    with torch.no_grad():
        for imgs, paths in loader:
            imgs = imgs.to(device)
            emb = backbone(imgs).view(imgs.size(0), -1)
            feats.append(emb.cpu().numpy())
            img_paths.extend(paths)
    if len(feats) == 0:
        return np.zeros((0,512)), []
    return np.vstack(feats), img_paths

# ---------------------------
# Main inference function
# ---------------------------
def run_inference(image_dir, model_dir, batch_size=8, output_csv="predictions.csv"):
    # Load backbone & projection head
    backbone = get_ssl_backbone(pretrained=False)
    proj_head = ProjectionHead().to(device)
    backbone.load_state_dict(torch.load(os.path.join(model_dir, "final_ssl_backbone.pth"), map_location=device))
    proj_head.load_state_dict(torch.load(os.path.join(model_dir, "final_ssl_proj_head.pth"), map_location=device))
    backbone.to(device).eval()
    proj_head.to(device).eval()

    # Load XGBoost model
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(os.path.join(model_dir, "final_xgb_model.json"))

    # Collect image paths
    valid_extensions = (".jpg", ".jpeg", ".png", ".heic", ".heif")
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)]
    if not image_paths:
        print("No valid images found in directory.")
        return

    # Extract embeddings
    embeddings, paths = extract_embeddings(image_paths, backbone, val_transform, batch_size=batch_size)

    # Predict Hb levels
    preds = xgb_model.predict(embeddings)

    # Save results
    df = pd.DataFrame({"Filename": paths, "Predicted_Hb": preds})
    df.to_csv(output_csv, index=False)
    print(f"✅ Predictions saved to {output_csv}")
    return df

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for Hb prediction using SSL + XGBoost")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images for inference")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing saved models")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--output_csv", type=str, default="predictions.csv", help="Output CSV file for predictions")
    args = parser.parse_args()

    run_inference(args.image_dir, args.model_dir, args.batch_size, args.output_csv)
