from config.config import DataConfig, ModelConfig, PathsConfig, TrainConfig
from data_manager.dataloaders import FrameDataModule
from Trainer import Trainer
from seed_utils import SEED, set_global_seed


def main():
    set_global_seed(SEED)
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
        epochs=60,
        folds=data_cfg.folds,
        use_scheduler=True,
        scheduler_name="cosine",
        cosine_t_max=60,
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

    data_module = FrameDataModule(data_cfg)
    train_loader, val_loader = data_module.get_loaders(fold_id=0)

    trainer = Trainer(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        path_cfg=paths_cfg,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    trainer._train_loop()


if __name__ == "__main__":
    main()
