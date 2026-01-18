import numpy as np
import pandas as pd
from PIL import Image
import random
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from seed_utils import SEED


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class FrameDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        path = row["path"]
        y = int(row["target"])

        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(y, dtype=torch.long)


class DataLoaderBuilder:
    def __init__(
        self,
        csv_path: str,
        seed: int = SEED,
        n_splits: int = 5,
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        self.csv_path = csv_path
        self.seed = seed
        self.n_splits = n_splits
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.df = self._load_df()

    def _load_df(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path).copy()

        required = {"path", "target", "patient_id"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing columns: {missing}")

        df["target"] = df["target"].astype(int)
        df["patient_id"] = df["patient_id"].astype(int)
        df = df.sort_values(["patient_id"]).reset_index(drop=True)

        return df

    def make_split(self):
        """
        Zwraca: (train_df, val_df, test_df)
        - test: jeden fold z SGKF(n_splits)
        - train/val: wewnętrzny split na train/val w obrębie train_val
        """
        df = self.df
        X = np.arange(len(df))
        y = df["target"].values
        g = df["patient_id"].values

        sgkf_outer = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

        # bierzemy pierwszy split jako "fold 0" (możesz iterować po wszystkich)
        train_val_idx, test_idx = next(sgkf_outer.split(X, y, g))

        # train/val w obrębie train_val
        tv_X = np.arange(len(train_val_idx))
        tv_y = y[train_val_idx]
        tv_g = g[train_val_idx]

        sgkf_inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=self.seed)
        train_rel, val_rel = next(sgkf_inner.split(tv_X, tv_y, tv_g))

        train_idx = train_val_idx[train_rel]
        val_idx = train_val_idx[val_rel]

        return df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]

    def make_loaders(self, transform_train=None, transform_eval=None):
        train_df, val_df, test_df = self.make_split()

        train_ds = FrameDataset(train_df, transform=transform_train)
        val_ds = FrameDataset(val_df, transform=transform_eval)
        test_ds = FrameDataset(test_df, transform=transform_eval)

        train_generator = torch.Generator().manual_seed(self.seed)
        eval_generator = torch.Generator().manual_seed(self.seed)
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            worker_init_fn=_seed_worker, generator=train_generator
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            worker_init_fn=_seed_worker, generator=eval_generator
        )
        test_loader = DataLoader(
            test_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            worker_init_fn=_seed_worker, generator=eval_generator
        )

        return train_loader, val_loader, test_loader
