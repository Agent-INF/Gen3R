#!/usr/bin/env python
"""
Training script for Gen3R Video Diffusion Model (Stage 2).

This script fine-tunes the video diffusion model (Wan 2.1) to jointly generate
appearance and geometry latent codes in a unified latent space.

Usage:
    python train_diffusion.py \
        --data_root /path/to/dataset \
        --pretrained_path ./checkpoints \
        --output_dir ./outputs/diffusion \
        --num_epochs 50 \
        --batch_size 1 \
        --learning_rate 1e-5

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
    DiffusionTrainer,
    get_dataloader,
)
from gen3r.training.trainer_diffusion import DiffusionTrainingConfig
from gen3r.pipeline import Gen3RPipeline

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
        description="Train Gen3R Video Diffusion Model (Stage 2)"
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
        "--pretrained_path",
        type=str,
        required=True,
        help="Path to pre-trained Gen3R checkpoint (containing transformer, geo_adapter, vggt, wan_vae)"
    )
    
    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/diffusion",
        help="Output directory for checkpoints and logs"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
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
        default=1e-5,
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
        default=500,
        help="Number of warmup steps"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping"
    )
    
    # Conditioning dropout arguments
    parser.add_argument(
        "--text_dropout_prob",
        type=float,
        default=0.1,
        help="Probability of dropping text condition"
    )
    parser.add_argument(
        "--camera_dropout_prob",
        type=float,
        default=0.1,
        help="Probability of dropping camera condition"
    )
    
    # Task sampling arguments
    parser.add_argument(
        "--task_1view_prob",
        type=float,
        default=0.4,
        help="Probability of 1-view task"
    )
    parser.add_argument(
        "--task_2view_prob",
        type=float,
        default=0.3,
        help="Probability of 2-view task"
    )
    parser.add_argument(
        "--task_allview_prob",
        type=float,
        default=0.3,
        help="Probability of all-view task"
    )
    
    # Data arguments
    parser.add_argument(
        "--num_frames",
        type=int,
        default=49,
        help="Number of frames per sequence"
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
        default=500,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=50,
        help="Log metrics every N steps"
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=250,
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
    
    # Validate task probabilities
    total_prob = args.task_1view_prob + args.task_2view_prob + args.task_allview_prob
    if abs(total_prob - 1.0) > 1e-6:
        logger.warning(f"Task probabilities sum to {total_prob}, normalizing...")
        args.task_1view_prob /= total_prob
        args.task_2view_prob /= total_prob
        args.task_allview_prob /= total_prob
    
    # Create training config
    config = DiffusionTrainingConfig(
        num_frames=args.num_frames,
        resolution=args.resolution,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        text_dropout_prob=args.text_dropout_prob,
        camera_dropout_prob=args.camera_dropout_prob,
        task_1view_prob=args.task_1view_prob,
        task_2view_prob=args.task_2view_prob,
        task_allview_prob=args.task_allview_prob,
        save_every=args.save_every,
        log_every=args.log_every,
        eval_every=args.eval_every,
        mixed_precision=args.mixed_precision,
        output_dir=output_dir,
        checkpoint_path=args.checkpoint_path,
    )
    
    # Load pre-trained pipeline
    logger.info(f"Loading pre-trained pipeline from {args.pretrained_path}")
    pipeline = Gen3RPipeline.from_pretrained(args.pretrained_path)
    
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
    trainer = DiffusionTrainer(
        config=config,
        transformer=pipeline.transformer,
        geo_adapter=pipeline.geo_adapter,
        vggt=pipeline.vggt,
        wan_vae=pipeline.wan_vae,
        text_encoder=pipeline.text_encoder,
        tokenizer=pipeline.tokenizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        scheduler=pipeline.scheduler,
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save final pipeline
    logger.info("Saving final pipeline...")
    pipeline.save_pretrained(os.path.join(output_dir, "final_pipeline"))
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
