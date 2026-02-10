#!/usr/bin/env python
"""
Training script for Gen3R Geometry Adapter (Stage 1).

This script trains the geometry adapter to map VGGT tokens to a latent space
aligned with the appearance latent space from Wan VAE.

Usage:
    python train_adapter.py \
        --data_root /path/to/dataset \
        --vggt_path /path/to/vggt \
        --wan_vae_path /path/to/wan_vae \
        --output_dir ./outputs/geometry_adapter \
        --num_epochs 100 \
        --batch_size 1 \
        --learning_rate 1e-4

Based on the paper: Gen3R: 3D Scene Generation Meets Feed-Forward Reconstruction
Paper: https://arxiv.org/abs/2601.04090
"""

import os
import sys
import argparse
import logging
from datetime import datetime

import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gen3r.training import (
    GeometryAdapterTrainer,
    get_dataloader,
)
from gen3r.training.trainer_adapter import TrainingConfig
from gen3r.models import GeometryAdapter, VGGT, AutoencoderKLWan

try:
    from accelerate.utils import set_seed
except ImportError:
    def set_seed(seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Gen3R Geometry Adapter (Stage 1)"
    )
    
    # Data arguments
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory containing the training dataset"
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="re10k",
        choices=["re10k", "dl3dv", "co3d", "wildrgbd", "tartanair", "mixed"],
        help="Type of dataset to use"
    )
    
    # Model arguments
    parser.add_argument(
        "--vggt_path",
        type=str,
        required=True,
        help="Path to pre-trained VGGT model"
    )
    parser.add_argument(
        "--wan_vae_path",
        type=str,
        required=True,
        help="Path to pre-trained Wan VAE model"
    )
    parser.add_argument(
        "--geo_adapter_path",
        type=str,
        default=None,
        help="Path to pre-trained geometry adapter (for fine-tuning)"
    )
    
    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/geometry_adapter",
        help="Output directory for checkpoints and logs"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=1000,
        help="Number of warmup steps"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping"
    )
    
    # Loss arguments
    parser.add_argument(
        "--lambda_rec",
        type=float,
        default=1.0,
        help="Weight for reconstruction loss"
    )
    parser.add_argument(
        "--lambda_kl",
        type=float,
        default=0.001,
        help="Weight for KL divergence loss"
    )
    
    # Data arguments
    parser.add_argument(
        "--num_frames",
        type=int,
        default=25,
        help="Number of frames per sequence (use 25 for initial training, 49 for fine-tuning)"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=560,
        help="Image resolution"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )
    
    # Checkpointing arguments
    parser.add_argument(
        "--save_every",
        type=int,
        default=1000,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=100,
        help="Log metrics every N steps"
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=500,
        help="Evaluate on validation set every N steps"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to resume training from checkpoint"
    )
    
    # Other arguments
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Add file handler for logging
    file_handler = logging.FileHandler(os.path.join(output_dir, "train.log"))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Arguments: {args}")
    logger.info(f"Output directory: {output_dir}")
    
    # Create training config
    config = TrainingConfig(
        num_frames=args.num_frames,
        resolution=args.resolution,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        lambda_rec=args.lambda_rec,
        lambda_kl=args.lambda_kl,
        save_every=args.save_every,
        log_every=args.log_every,
        eval_every=args.eval_every,
        mixed_precision=args.mixed_precision,
        output_dir=output_dir,
        checkpoint_path=args.checkpoint_path,
    )
    
    # Load models
    logger.info("Loading models...")
    
    # Load or initialize geometry adapter
    if args.geo_adapter_path is not None:
        logger.info(f"Loading geometry adapter from {args.geo_adapter_path}")
        geometry_adapter = GeometryAdapter.from_pretrained(args.geo_adapter_path)
    else:
        logger.info("Initializing new geometry adapter")
        geometry_adapter = GeometryAdapter()
    
    # Load VGGT
    logger.info(f"Loading VGGT from {args.vggt_path}")
    vggt = VGGT.from_pretrained(args.vggt_path)
    
    # Load Wan VAE
    logger.info(f"Loading Wan VAE from {args.wan_vae_path}")
    wan_vae = AutoencoderKLWan.from_pretrained(args.wan_vae_path)
    
    # Create dataloaders
    logger.info(f"Creating dataloaders from {args.data_root}")
    
    train_dataloader = get_dataloader(
        data_root=args.data_root,
        split="train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_frames=args.num_frames,
        resolution=(args.resolution, args.resolution),
        dataset_type=args.dataset_type,
    )
    
    # Optional validation dataloader
    val_dataloader = None
    val_data_path = os.path.join(args.data_root, "val")
    if os.path.exists(val_data_path):
        val_dataloader = get_dataloader(
            data_root=args.data_root,
            split="val",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_frames=args.num_frames,
            resolution=(args.resolution, args.resolution),
            dataset_type=args.dataset_type,
            shuffle=False,
        )
    
    logger.info(f"Training samples: {len(train_dataloader.dataset)}")
    if val_dataloader is not None:
        logger.info(f"Validation samples: {len(val_dataloader.dataset)}")
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = GeometryAdapterTrainer(
        config=config,
        geometry_adapter=geometry_adapter,
        vggt=vggt,
        wan_vae=wan_vae,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
