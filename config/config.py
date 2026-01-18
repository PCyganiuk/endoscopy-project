from dataclasses import dataclass

@dataclass
class ModelConfig:
    model_name: str
    multi_label: bool #= False
    num_classes: int
    learning_rate: int

    opt_name: str
    weight_decay: int



@dataclass
class TrainConfig:
    epochs: int #= 60
    folds: int #= 5

    use_scheduler: bool #= True
    scheduler_name: str #= "cosine"
    cosine_t_max: int #= epochs

    step_size_lr: int
    gamma: float

    mode: str
    factor: float
    patiance: int

@dataclass
class DataConfig:
    csv_path: str
    frames_root: str
    image_size: tuple[int, int] = (224, 224)
    batch_size: int = 128
    num_workers: int = 4
    folds: int = 5
    seed: int = 190836
    pin_memory: bool = True

@dataclass
class PathsConfig:
    model_path: str
    metrics_json_path: str
    
    



