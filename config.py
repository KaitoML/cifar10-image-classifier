from dataclasses import dataclass

@dataclass
class Config:
    device: str
    n_epochs: int
    batch_size: int
    scheduler_patience: int
    early_stopping_patience: int
    best_params_path: str
    l2: float
    scheduler_factor: float
    image_resize: tuple
    trainable_layers_with_lr: dict


