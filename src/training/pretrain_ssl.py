import os
import random
import time
from datetime import datetime
import joblib

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pillow_heif
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Register HEIF opener (if using HEIC/HEIF images)
pillow_heif.register_heif_opener()

# ---------------------------
# Configuration / Reproducibility
# ---------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")







# ---------------------------
# Transforms
# ---------------------------
ssl_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    # transforms.ColorJitter(0.4,0.4,0.4,0.1), => May be too strong for hb predictions
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.RandomResizedCrop(224, scale=(0.8,1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ---------------------------
# ResNet Backbone + Projection Head
# ---------------------------
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

def get_ssl_backbone(pretrained=True):
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    return backbone.to(device)

# ---------------------------
# NT-Xent Loss
# ---------------------------
def nt_xent_loss(z_i, z_j, temperature=0.5):
    z_i = nn.functional.normalize(z_i, dim=1)
    z_j = nn.functional.normalize(z_j, dim=1)
    batch_size = z_i.size(0)
    representations = torch.cat([z_i, z_j], dim=0)
    similarity_matrix = torch.matmul(representations, representations.T) / temperature
    mask = torch.eye(2*batch_size, device=z_i.device).bool()
    similarity_matrix = similarity_matrix.masked_fill(mask, -1e9)
    labels = torch.arange(batch_size, device=z_i.device)
    labels = torch.cat([labels + batch_size, labels], dim=0)
    return nn.CrossEntropyLoss()(similarity_matrix, labels)

# ---------------------------
# SSL Pretraining (supports unlabeled or labeled)
# ---------------------------
def pretrain_ssl(labelled_df=None,
                          unlabelled_dir=None,
                          transform=ssl_transform,
                          epochs=20,
                          batch_size=8,
                          lr=1e-3,
                          num_workers=2,
                          run_dir=None):

    dataset = CombinedImageDataset(labelled_df=labelled_df,
                                   unlabelled_dir=unlabelled_dir,
                                   transform=transform,
                                   n_views=2)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    backbone = get_ssl_backbone(pretrained=True)
    proj_head = ProjectionHead().to(device)
    optimizer = torch.optim.Adam(list(backbone.parameters()) + list(proj_head.parameters()), lr=lr)

    backbone.train()
    proj_head.train()
    ssl_losses = []

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            views, target, is_labeled = batch
            v1, v2 = views[:,0].to(device), views[:,1].to(device)

            feats1 = backbone(v1).view(v1.size(0), -1)
            feats2 = backbone(v2).view(v2.size(0), -1)
            z1 = proj_head(feats1)
            z2 = proj_head(feats2)
            loss = nt_xent_loss(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader) if len(loader) > 0 else float("nan")
        ssl_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - SSL Loss: {avg_loss:.4f}")

    if run_dir:
        torch.save(backbone.state_dict(), os.path.join(run_dir, "ssl_backbone_state_dict.pth"))
        torch.save(proj_head.state_dict(), os.path.join(run_dir, "ssl_projection_head_state_dict.pth"))
        pd.DataFrame({"ssl_loss": ssl_losses}).to_csv(os.path.join(run_dir, "ssl_loss_history.csv"), index=False)

    return backbone, proj_head, ssl_losses

