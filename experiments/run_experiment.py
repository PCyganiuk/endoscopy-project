import json
from datetime import datetime

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from config.config import ModelConfig, TrainConfig
from Trainer import Trainer
from sklearn.metrics import confusion_matrix


def build_cifar10_loaders(batch_size: int = 32, num_workers: int = 2, train_size: int = 5000, val_size: int = 1000):
    """CIFAR-10 (10 klas) + transformy pod modele ImageNet (224x224)."""
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    full_train = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)

    # Mały split do sanity-check (żeby szybko odpalić).
    total = train_size + val_size
    if total > len(full_train):
        raise ValueError(f"train_size+val_size={total} > len(CIFAR10)={len(full_train)}")

    subset, _ = random_split(full_train, [total, len(full_train) - total], generator=torch.Generator().manual_seed(42))
    train_ds, val_ds = random_split(subset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return train_loader, val_loader


def run():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1) DataLoadery
    train_loader, val_loader = build_cifar10_loaders(batch_size=32, num_workers=2, train_size=5000, val_size=1000)

    # 2) Konfiguracje (ważne: CIFAR10 => num_classes=10, multi_label=False)
    model_cfg = ModelConfig(
        model_name="mobilenetv2",    # szybciej niż resnet50
        num_classes=10,
        learning_rate=1e-3,
        opt_name="adamw",
        weight_decay=1e-4,
        multi_label=False,
        use_focal_loss=False,
    )

    train_cfg = TrainConfig(
        epochs=3,
        folds = 1,
        use_scheduler=True,
        scheduler_name="cosine",
        cosine_t_max=3,

        # pola wymagane w TrainConfig (nawet jeśli akurat nieużywane przy cosine):
        step_size_lr=5,
        gamma=0.1,
        mode="min",
        factor=0.1,
        patiance=2,
    )

    # 3) Trener
    trainer = Trainer(model_cfg=model_cfg, train_cfg=train_cfg, train_loader=train_loader, val_loader=val_loader)

    # 4) Trening
    trainer._train_loop()

    # 5) Po treningu: zbierz predykcje/targety na walidacji (do confusion matrix)
    trainer.model.eval()
    all_logits = []
    all_targets = []
    device = trainer.device

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            logits = trainer.model(x)
            all_logits.append(logits.detach().cpu())
            all_targets.append(y.detach().cpu())

    logits_epoch = torch.cat(all_logits, dim=0)
    targets_epoch = torch.cat(all_targets, dim=0)

    # zapis "finalnych" predykcji/targetów (ostatnia epoka / finalny model)
    trainer.calculate_metrics.get_predictions_targets(logits=logits_epoch, targets=targets_epoch, epoch=train_cfg.epochs - 1)

    preds_np = trainer.calculate_metrics.get_history()["predictions"][-1]
    targets_np = trainer.calculate_metrics.get_history()["targets"][-1]
    cm = confusion_matrix(targets_np, preds_np)

    # 6) Zapis metryk do JSON
    out = {
        "model_cfg": model_cfg.__dict__,
        "train_cfg": train_cfg.__dict__,
        "metrics_history": trainer.calculate_metrics.get_history(),
        "confusion_matrix": cm.tolist(),
    }

    metrics_path = f"metrics_{timestamp}.json"
    with open(metrics_path, "w") as f:
        json.dump(out, f, indent=2)

    # 7) TensorBoard: logowanie metryk (po fakcie, z historii)
    writer = SummaryWriter(log_dir=f"runs/sanity_{timestamp}")
    hist = trainer.calculate_metrics.get_history()

    for name, values in hist.items():
        # predictions/targets to tablice (nie listy po epokach) — pomijamy w skalarach
        if name in ("predictions", "targets"):
            continue
        for ep, v in enumerate(values, start=1):
            writer.add_scalar(f"metrics/{name}", float(v), ep)

    writer.flush()
    writer.close()

    print(f"\n Zapisano: {metrics_path}")
    print(" TensorBoard: uruchom `tensorboard --logdir runs`")
    print(" Confusion matrix shape:", cm.shape)


if __name__ == "__main__":
    run()
