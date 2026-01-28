"""
Geometry Adapter Trainer for Gen3R.

This module implements Stage 1 of Gen3R training:
Training the geometry adapter to map VGGT tokens to geometry latent space
that aligns with the appearance latent space from Wan VAE.
"""

import os
import math
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader
from einops import rearrange
from tqdm import tqdm

try:
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False

from ..models import GeometryAdapter, VGGT, AutoencoderKLWan
from .losses import GeometryAdapterLoss


logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for geometry adapter training."""
    # Data
    num_frames: int = 25  # Initial frames, can increase to 49 for fine-tuning
    resolution: int = 560
    batch_size: int = 1
    
    # Training
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Loss weights
    lambda_rec: float = 1.0
    lambda_kl: float = 0.001
    
    # Checkpointing
    save_every: int = 1000
    log_every: int = 100
    eval_every: int = 500
    
    # Mixed precision
    mixed_precision: str = "bf16"  # "no", "fp16", "bf16"
    
    # Paths
    output_dir: str = "./outputs/geometry_adapter"
    checkpoint_path: Optional[str] = None


class GeometryAdapterTrainer:
    """
    Trainer for the Geometry Adapter (Stage 1 of Gen3R training).
    
    This trainer:
    1. Freezes VGGT and Wan VAE weights
    2. Trains only the geometry adapter (encoder E_adp and decoder D_adp)
    3. Optimizes reconstruction loss and distribution alignment loss
    
    Args:
        config: Training configuration
        geometry_adapter: Geometry adapter model
        vggt: Pre-trained VGGT model (frozen)
        wan_vae: Pre-trained Wan VAE (frozen)
        train_dataloader: Training dataloader
        val_dataloader: Optional validation dataloader
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        geometry_adapter: GeometryAdapter,
        vggt: VGGT,
        wan_vae: AutoencoderKLWan,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
    ):
        self.config = config
        self.geometry_adapter = geometry_adapter
        self.vggt = vggt
        self.wan_vae = wan_vae
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Freeze VGGT and Wan VAE
        self._freeze_model(self.vggt)
        self._freeze_model(self.wan_vae)
        
        # Initialize loss function
        self.loss_fn = GeometryAdapterLoss(
            lambda_rec=config.lambda_rec,
            lambda_kl=config.lambda_kl,
        )
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.geometry_adapter.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Initialize learning rate scheduler
        total_steps = config.num_epochs * len(train_dataloader)
        self.scheduler = self._get_scheduler(total_steps)
        
        # Initialize accelerator if available
        if ACCELERATE_AVAILABLE:
            self.accelerator = Accelerator(
                mixed_precision=config.mixed_precision,
                gradient_accumulation_steps=1,
            )
            (
                self.geometry_adapter,
                self.vggt,
                self.wan_vae,
                self.optimizer,
                self.train_dataloader,
                self.scheduler,
            ) = self.accelerator.prepare(
                self.geometry_adapter,
                self.vggt,
                self.wan_vae,
                self.optimizer,
                self.train_dataloader,
                self.scheduler,
            )
            self.device = self.accelerator.device
        else:
            self.accelerator = None
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.geometry_adapter = self.geometry_adapter.to(self.device)
            self.vggt = self.vggt.to(self.device)
            self.wan_vae = self.wan_vae.to(self.device)
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
    def _freeze_model(self, model: nn.Module) -> None:
        """Freeze all parameters of a model."""
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
    
    def _get_scheduler(self, total_steps: int):
        """Get learning rate scheduler with warmup."""
        warmup_steps = self.config.warmup_steps
        
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    @torch.no_grad()
    def extract_vggt_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract VGGT tokens from input images.
        
        Args:
            images: Input images [B, F, 3, H, W] in range [0, 1]
            
        Returns:
            VGGT tokens [B, C, F, H', W']
        """
        # VGGT expects [B, F, 3, H, W]
        B, F, C, H, W = images.shape
        
        # Get aggregated tokens from VGGT
        aggregated_tokens_list, ps_idx = self.vggt.aggregator(images)
        
        # Use tokens from intermediate layers
        intermediate_idx = self.vggt.depth_head.intermediate_layer_idx
        tokens = [aggregated_tokens_list[i] for i in intermediate_idx]
        
        # Concatenate tokens from different layers
        # Each token has shape [B*F, num_patches, embed_dim]
        concat_tokens = torch.cat(tokens, dim=-1)  # [B*F, num_patches, total_dim]
        
        # Reshape to [B, F, H', W', C'] format for adapter
        patch_size = self.vggt.aggregator.patch_size
        h_patches = H // patch_size
        w_patches = W // patch_size
        
        tokens_reshaped = concat_tokens[:, ps_idx:, :].reshape(
            B, F, h_patches, w_patches, -1
        )  # [B, F, H', W', C']
        
        # Rearrange to [B, C', F, H', W']
        tokens_reshaped = rearrange(tokens_reshaped, 'b f h w c -> b c f h w')
        
        return tokens_reshaped
    
    @torch.no_grad()
    def extract_appearance_latent(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract appearance latent codes using Wan VAE.
        
        Args:
            images: Input images [B, F, 3, H, W] in range [0, 1]
            
        Returns:
            Appearance latent [B, C, f, h, w]
        """
        # Convert to Wan VAE input format
        images = rearrange((images * 2 - 1).clamp(-1, 1), 'b f c h w -> b c f h w')
        
        # Encode with Wan VAE
        latent = self.wan_vae.encode(images).latent_dist.sample()
        
        return latent
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            batch: Batch of training data
            
        Returns:
            Dictionary of loss values
        """
        self.geometry_adapter.train()
        
        # Get input images
        images = batch["frames"].to(self.device)  # [B, F, 3, H, W]
        
        # Extract VGGT tokens (frozen)
        vggt_tokens = self.extract_vggt_tokens(images)  # [B, C, F, H', W']
        
        # Extract appearance latent (frozen)
        appearance_latent = self.extract_appearance_latent(images)  # [B, C, f, h, w]
        
        # Forward through geometry adapter
        geometry_latent = self.geometry_adapter.encode(vggt_tokens).latent_dist
        geometry_mu = geometry_latent.mean
        geometry_logvar = geometry_latent.logvar
        geometry_samples = geometry_latent.sample()
        
        # Decode back to VGGT tokens
        reconstructed_tokens = self.geometry_adapter.decode(geometry_samples).sample
        
        # Compute losses
        losses = self.loss_fn(
            reconstructed_tokens=reconstructed_tokens,
            original_tokens=vggt_tokens,
            geometry_latent=geometry_samples,
            appearance_latent=appearance_latent,
            geometry_mu=geometry_mu,
            geometry_logvar=geometry_logvar,
        )
        
        # Backward pass
        total_loss = losses['total_loss']
        
        if self.accelerator is not None:
            self.accelerator.backward(total_loss)
        else:
            total_loss.backward()
        
        # Gradient clipping
        if self.config.max_grad_norm > 0:
            if self.accelerator is not None:
                self.accelerator.clip_grad_norm_(
                    self.geometry_adapter.parameters(),
                    self.config.max_grad_norm
                )
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.geometry_adapter.parameters(),
                    self.config.max_grad_norm
                )
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        # Return scalar loss values
        return {k: v.item() for k, v in losses.items()}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation loop."""
        if self.val_dataloader is None:
            return {}
        
        self.geometry_adapter.eval()
        
        total_losses = {}
        num_batches = 0
        
        for batch in self.val_dataloader:
            images = batch["frames"].to(self.device)
            
            # Extract VGGT tokens
            vggt_tokens = self.extract_vggt_tokens(images)
            
            # Extract appearance latent
            appearance_latent = self.extract_appearance_latent(images)
            
            # Forward through geometry adapter
            geometry_latent = self.geometry_adapter.encode(vggt_tokens).latent_dist
            geometry_samples = geometry_latent.sample()
            
            # Decode
            reconstructed_tokens = self.geometry_adapter.decode(geometry_samples).sample
            
            # Compute losses
            losses = self.loss_fn(
                reconstructed_tokens=reconstructed_tokens,
                original_tokens=vggt_tokens,
                geometry_latent=geometry_samples,
                appearance_latent=appearance_latent,
                geometry_mu=geometry_latent.mean,
                geometry_logvar=geometry_latent.logvar,
            )
            
            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0.0) + v.item()
            num_batches += 1
        
        # Average losses
        return {k: v / num_batches for k, v in total_losses.items()}
    
    def save_checkpoint(self, path: Optional[str] = None) -> None:
        """Save training checkpoint."""
        if path is None:
            path = os.path.join(
                self.config.output_dir,
                f"checkpoint-{self.global_step}.pt"
            )
        
        # Get model state dict
        if self.accelerator is not None:
            model_state = self.accelerator.unwrap_model(self.geometry_adapter).state_dict()
        else:
            model_state = self.geometry_adapter.state_dict()
        
        checkpoint = {
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "config": self.config,
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
        
        # Save geometry adapter in diffusers-compatible format
        adapter_path = os.path.join(self.config.output_dir, f"geo_adapter-{self.global_step}")
        os.makedirs(adapter_path, exist_ok=True)
        
        if self.accelerator is not None:
            self.accelerator.unwrap_model(self.geometry_adapter).save_pretrained_safetensors(adapter_path)
        else:
            self.geometry_adapter.save_pretrained_safetensors(adapter_path)
    
    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        if self.accelerator is not None:
            self.accelerator.unwrap_model(self.geometry_adapter).load_state_dict(
                checkpoint["model_state_dict"]
            )
        else:
            self.geometry_adapter.load_state_dict(checkpoint["model_state_dict"])
        
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.current_epoch = checkpoint["current_epoch"]
        
        logger.info(f"Checkpoint loaded from {path}")
    
    def train(self) -> None:
        """Run the full training loop."""
        logger.info("Starting geometry adapter training...")
        logger.info(f"Config: {self.config}")
        
        # Load checkpoint if specified
        if self.config.checkpoint_path is not None:
            self.load_checkpoint(self.config.checkpoint_path)
        
        # Training loop
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            epoch_losses = {}
            
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}/{self.config.num_epochs}",
                disable=self.accelerator is not None and not self.accelerator.is_main_process,
            )
            
            for batch in progress_bar:
                # Training step
                losses = self.train_step(batch)
                self.global_step += 1
                
                # Accumulate losses
                for k, v in losses.items():
                    epoch_losses[k] = epoch_losses.get(k, 0.0) + v
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{losses['total_loss']:.4f}",
                    "rec": f"{losses['rec_loss']:.4f}",
                    "kl": f"{losses['kl_loss']:.4f}",
                })
                
                # Logging
                if self.global_step % self.config.log_every == 0:
                    logger.info(
                        f"Step {self.global_step}: "
                        f"loss={losses['total_loss']:.4f}, "
                        f"rec={losses['rec_loss']:.4f}, "
                        f"kl={losses['kl_loss']:.4f}"
                    )
                
                # Validation
                if self.global_step % self.config.eval_every == 0:
                    val_losses = self.validate()
                    if val_losses:
                        logger.info(f"Validation: {val_losses}")
                
                # Checkpointing
                if self.global_step % self.config.save_every == 0:
                    self.save_checkpoint()
            
            # End of epoch
            avg_losses = {k: v / len(self.train_dataloader) for k, v in epoch_losses.items()}
            logger.info(f"Epoch {epoch + 1} average losses: {avg_losses}")
        
        # Final checkpoint
        self.save_checkpoint(os.path.join(self.config.output_dir, "final_checkpoint.pt"))
        logger.info("Training completed!")


def train_geometry_adapter(
    data_root: str,
    vggt_path: str,
    wan_vae_path: str,
    output_dir: str = "./outputs/geometry_adapter",
    **kwargs
) -> GeometryAdapter:
    """
    Convenience function to train the geometry adapter.
    
    Args:
        data_root: Path to training data
        vggt_path: Path to pre-trained VGGT model
        wan_vae_path: Path to pre-trained Wan VAE
        output_dir: Output directory for checkpoints
        **kwargs: Additional training configuration
        
    Returns:
        Trained geometry adapter model
    """
    from .datasets import get_dataloader
    
    # Create config
    config = TrainingConfig(output_dir=output_dir, **kwargs)
    
    # Load models
    geometry_adapter = GeometryAdapter()
    vggt = VGGT.from_pretrained(vggt_path)
    wan_vae = AutoencoderKLWan.from_pretrained(wan_vae_path)
    
    # Create dataloader
    train_dataloader = get_dataloader(
        data_root=data_root,
        split="train",
        batch_size=config.batch_size,
        num_frames=config.num_frames,
        resolution=(config.resolution, config.resolution),
    )
    
    # Create trainer
    trainer = GeometryAdapterTrainer(
        config=config,
        geometry_adapter=geometry_adapter,
        vggt=vggt,
        wan_vae=wan_vae,
        train_dataloader=train_dataloader,
    )
    
    # Train
    trainer.train()
    
    return geometry_adapter
