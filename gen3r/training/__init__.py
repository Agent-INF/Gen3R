# Training modules for Gen3R
# Based on the paper: Gen3R: 3D Scene Generation Meets Feed-Forward Reconstruction

from .datasets import MultiViewDataset, get_dataloader
from .losses import (
    GeometryAdapterLoss,
    reconstruction_loss,
    distribution_alignment_loss,
)
from .trainer_adapter import GeometryAdapterTrainer
from .trainer_diffusion import DiffusionTrainer

__all__ = [
    "MultiViewDataset",
    "get_dataloader",
    "GeometryAdapterLoss",
    "reconstruction_loss",
    "distribution_alignment_loss",
    "GeometryAdapterTrainer",
    "DiffusionTrainer",
]
