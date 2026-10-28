"""Model and training configuration: the single source of truth for hyperparameters."""

from dataclasses import asdict, dataclass


@dataclass
class ViTConfig:
    img_size: int = 28
    patch_size: int = 7
    num_channels: int = 1
    num_heads: int = 1
    embed_dim: int = 16
    mlp_dim: int = 16
    transformer_units: int = 1
    num_classes: int = 10

    @property
    def num_patches(self) -> int:
        return (self.img_size // self.patch_size) ** 2

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainConfig:
    epochs: int = 5
    batch_size: int = 64
    lr: float = 0.01
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)
