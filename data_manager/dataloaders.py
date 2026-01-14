import torch
from torch.utils.data import DataLoader
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from torchvision import transforms
from src.config.config import PathsConfig, DataConfig
import numpy as np

df = pd.read_csv("/home/blade/Desktop/studia/research/merged_galar.csv")
df = df.copy()
df["frame"] = df["frame"].astype(int)
df["target"] = df["target"].astype(int)
df["patient_id"] = df["patient_id"].astype(int)

df = df.sort_values(["patient_id", "frame"]).reset_index(drop=True)


X = df.index.values
y = df["target"].values
groups = df["patient_id"].values

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
folds = [(tr, va) for tr, va in sgkf.split(X, y, groups)]


def frame_to_path(root, patient_id, frame):
    # DOSTOSUJ DO SWOJEJ STRUKTURY:
    return os.path.join(root, str(patient_id), f"frame_{frame:06d}.PNG")

class FrameDataset(Dataset):
    def __init__(self, df, root, transform=None):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        path = frame_to_path(self.root, row.patient_id, row.frame)
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        y = torch.tensor(int(row.target), dtype=torch.long)
        return img, y


fold_id = 0
tr_idx, va_idx = folds[fold_id]

train_df = df.iloc[tr_idx]
val_df   = df.iloc[va_idx]

transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),          
])


train_ds = FrameDataset(train_df, root="/home/blade/ml/studia/zpb", transform=transform)
val_ds   = FrameDataset(val_df, root="/home/blade/ml/studia/zpb", transform=transform)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)


 
def _make_folds(df: pd.DataFrame, n_splits: int, seed: int):
        X = df.index.values
        y = df["target"].values

        groups = df["patient_id"].values
        sgkf_train_test = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        train_val_idxs, test_idxs = next(sgkf_train_test.split(X, y, groups))

        sgkf_train_val = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
        tv_X = np.arange(len(train_val_idxs))
        tv_y = y[train_val_idxs]
        tv_g = groups[train_val_idxs]
        train_rel, val_rel = next(sgkf_train_val.split(tv_X, tv_y, tv_g))
        train_idxs = train_val_idxs[train_rel]
        val_idxs   = train_val_idxs[val_rel]

        return test_idxs, train_idxs, val_idxs

a, b, c = _make_folds(df=df, n_splits=5, seed=42)        
print(a)
print("--")
        
print(b)
print("--")
        
print(c)
print("--")