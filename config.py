from dataclasses import dataclass

@dataclass
class ModelConfig:
    model_name: str
    multi_label: bool = False
    num_classes: int
    learning_rate: int

    opt_name: str
    weight_decay: int

    use_focal_loss: bool = False


@dataclass
class TrainConfig:
    epochs: int = 60
    folds: int = 5

    use_scheduler: bool = True
    scheduler_name: str = "cosine"
    cosine_t_max: int = epochs

    step_size_lr: int
    gamma: float

    mode: str
    factor: float
    patiance: int
    
@dataclass
class PathsConfig:
    root_dir_path: str
    metrics_path: str
    data_path: str
    galar_csv_path: str

@dataclass
class DataConfig:
    image_size = (224, 224)
    batch_size = 256
    num_workers=8
    folds=5
    seed=190836
    pin_memory=True




