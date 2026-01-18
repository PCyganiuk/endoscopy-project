from torchvision import transforms
from data_manager.dataloaders import DataLoaderBuilder
from config.config import DataConfig, ModelConfig, PathsConfig, TrainConfig
from Trainer import Trainer
from seed_utils import SEED, set_global_seed



"""Krok 1: Ustawienie seed'a"""
set_global_seed(SEED)

"""Krok 2: Ustawienie parametrów konfiguracyjnych"""

data_cfg = DataConfig(
    csv_path="merged_galar.csv",
    frames_root="/home/blade/ml/studia/zpb",
    image_size=(224, 224),
    batch_size=128,
    num_workers=4,
    folds=5,
    seed=SEED,
    pin_memory=True,
)

model_cfg = ModelConfig(
    model_name="resnet18",
    multi_label=False,
    num_classes=2,
    learning_rate=1e-3,
    opt_name="adam",
    weight_decay=1e-4,
    use_focal_loss=False,
)

train_cfg = TrainConfig(
    epochs=1,
    folds=data_cfg.folds,
    use_scheduler=True,
    scheduler_name="cosine",
    cosine_t_max=1,
    step_size_lr=10,
    gamma=0.1,
    mode="min",
    factor=0.5,
    patiance=5,
)

paths_cfg = PathsConfig(
    model_path="checkpoints",
    metrics_json_path="metrics.jsonl",
)

"""Krok 3: Silna augmentacja danch"""
import torch
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode

heavy_endoscopy = v2.Compose([

    v2.ToTensor(),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.3),
    v2.RandomApply([v2.RandomRotation(degrees=25, expand=False, fill=0)], p=0.7),
    v2.RandomApply([
        v2.RandomAffine(
            degrees=20,
            translate=(0.08, 0.08),
            scale=(0.85, 1.15),
            shear=(-10, 10),
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )
    ], p=0.8),
    v2.RandomPerspective(distortion_scale=0.6, p=0.6, fill=0),  
    v2.RandomApply([
        v2.ElasticTransform(alpha=80.0, sigma=6.0, interpolation=InterpolationMode.BILINEAR)
    ], p=0.35), 
    v2.ColorJitter(brightness=0.45, contrast=0.45, saturation=0.25, hue=0.08), 
    v2.RandomPhotometricDistort(
        brightness=(0.7, 1.3),
        contrast=(0.6, 1.6),
        saturation=(0.6, 1.6),
        hue=(-0.08, 0.08),
        p=0.7,
    ), 
    v2.RandomApply([v2.GaussianBlur(kernel_size=9, sigma=(0.1, 2.5))], p=0.55),  
    v2.ToDtype(torch.float32, scale=True),
    v2.RandomApply([v2.GaussianNoise(mean=0.0, sigma=0.06, clip=True)], p=0.6),  

])

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

transform_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

"""Krok 4: Budowa DataLoaderow"""

builder = DataLoaderBuilder(
    csv_path="",
    seed=42,
    n_splits=5,
    batch_size=128,
    num_workers=4,
)
train_loader, val_loader, test_loader = builder.make_loaders(
    transform_train=transform_train,
    transform_eval=transform_eval,
)

"""Krok 5: Budowa trenera"""

trainer = Trainer(
    model_cfg=model_cfg,
    train_cfg=train_cfg,
    path_cfg=paths_cfg,
    train_loader=train_loader,
    val_loader=val_loader,
)

"""Krok 6: Petla treningowa"""
trainer._train_loop()






