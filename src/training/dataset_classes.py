# ---------------------------
# Dataset Classes
# ---------------------------
from torch.utils.data import Dataset
from PIL import Image
import torch
import os

class HbImageDataset(Dataset):
    """Labeled dataset for supervised + SSL training"""
    def __init__(self, df, transform=None, path_col="Filename", target_col="Hb", n_views=2):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.path_col = path_col
        self.target_col = target_col
        self.n_views = n_views

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row[self.path_col]).convert("RGB")
        if self.n_views == 1:
            return self.transform(img).unsqueeze(0), torch.tensor(row[self.target_col], dtype=torch.float32)
        else:
            views = [self.transform(img) for _ in range(self.n_views)]
            return torch.stack(views), torch.tensor(row[self.target_col], dtype=torch.float32)

class UnlabelledImageDataset(Dataset):
    """Unlabeled dataset for SSL pretraining, supports subfolders"""
    def __init__(self, root_dir, transform=None, n_views=2):
        self.image_paths = []
        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".heic", ".heif")):
                    self.image_paths.append(os.path.join(dirpath, fname))
        self.transform = transform
        self.n_views = n_views

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.n_views == 1:
            return self.transform(img).unsqueeze(0)
        else:
            views = [self.transform(img) for _ in range(self.n_views)]
            return torch.stack(views)

class CombinedImageDataset(Dataset):
    """
    Combines a labeled DataFrame dataset and an unlabelled image directory for SSL.
    Supports multi-view transformations for contrastive learning.
    Returns (views, target, is_labeled).
    Logs the number of labeled and unlabeled images loaded.
    """
    def __init__(self, labelled_df=None, unlabelled_dir=None, transform=None, path_col="Filename", target_col="Hb", n_views=2):
        self.labelled_dataset = None
        self.unlabelled_dataset = None

        if labelled_df is not None:
            self.labelled_dataset = HbImageDataset(
                labelled_df, transform=transform, path_col=path_col, target_col=target_col, n_views=n_views
            )
            print(f"[INFO] Loaded {len(self.labelled_dataset)} labeled images for SSL pretraining.")

        if unlabelled_dir is not None:
            self.unlabelled_dataset = UnlabelledImageDataset(
                unlabelled_dir, transform=transform, n_views=n_views
            )
            print(f"[INFO] Loaded {len(self.unlabelled_dataset)} unlabeled images for SSL pretraining.")

        # Compute total length
        self.labelled_len = len(self.labelled_dataset) if self.labelled_dataset else 0
        self.unlabelled_len = len(self.unlabelled_dataset) if self.unlabelled_dataset else 0
        self.total_len = self.labelled_len + self.unlabelled_len
        print(f"[INFO] Total images in combined dataset: {self.total_len}")

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if self.labelled_dataset and idx < self.labelled_len:
            views, target = self.labelled_dataset[idx]
            return views, target, True  # True indicates labeled
        else:
            unlabelled_idx = idx - self.labelled_len
            views = self.unlabelled_dataset[unlabelled_idx]
            dummy_target = torch.tensor(-1.0)  # Dummy target for unlabeled
            return views, dummy_target, False  # False indicates unlabelled