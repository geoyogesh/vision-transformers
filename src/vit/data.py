"""MNIST datasets and loaders shared by training and inference."""

import torch
import torchvision
import torchvision.transforms as transforms

DEFAULT_DATA_DIR = "data"


def transform():
    return transforms.Compose([transforms.ToTensor()])


def mnist(train: bool, root: str = DEFAULT_DATA_DIR, with_transform: bool = True):
    return torchvision.datasets.MNIST(
        root, train=train, download=True, transform=transform() if with_transform else None
    )


def loader(
    train: bool, batch_size: int, root: str = DEFAULT_DATA_DIR
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(mnist(train, root), batch_size=batch_size, shuffle=train)
