from dataclasses import dataclass

@dataclass
class ModelConfig:
    model_name: str
    multi_label: bool #= False
    num_classes: int
    learning_rate: int

    opt_name: str
    weight_decay: int

    use_focal_loss: bool #= False


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
    





