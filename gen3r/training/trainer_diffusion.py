"""
Diffusion Model Trainer for Gen3R.

This module implements Stage 2 of Gen3R training:
Fine-tuning the video diffusion model (Wan2.1) to jointly generate
appearance and geometry latent codes.
"""

import os
import math
import logging
import random
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from einops import rearrange
from tqdm import tqdm

try:
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False

from ..models import (
    GeometryAdapter, 
    VGGT, 
    AutoencoderKLWan,
    WanTransformer3DModel,
    WanT5EncoderModel,
    AutoTokenizer,
)
from .losses import DiffusionLoss


logger = logging.getLogger(__name__)


@dataclass
class DiffusionTrainingConfig:
    """Configuration for diffusion model training."""
    # Data
    num_frames: int = 49
    resolution: int = 560
    batch_size: int = 1
    
    # Training
    num_epochs: int = 50
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    
    # Conditioning dropout rates for flexibility
    text_dropout_prob: float = 0.1
    camera_dropout_prob: float = 0.1
    
    # Task sampling probabilities
    task_1view_prob: float = 0.4
    task_2view_prob: float = 0.3
    task_allview_prob: float = 0.3
    
    # Checkpointing
    save_every: int = 500
    log_every: int = 50
    eval_every: int = 250
    
    # Mixed precision
    mixed_precision: str = "bf16"
    
    # Paths
    output_dir: str = "./outputs/diffusion"
    checkpoint_path: Optional[str] = None


class DiffusionTrainer:
    """
    Trainer for the Video Diffusion Model (Stage 2 of Gen3R training).
    
    This trainer fine-tunes a pre-trained video diffusion model to jointly
    generate appearance latent codes (A) and geometry latent codes (G)
    by operating on the concatenated latent space Z = [A; G].
    
    Args:
        config: Training configuration
        transformer: Transformer model (Wan Transformer)
        geo_adapter: Pre-trained geometry adapter (frozen)
        vggt: Pre-trained VGGT model (frozen)
        wan_vae: Pre-trained Wan VAE (frozen)
        text_encoder: Text encoder model
        tokenizer: Tokenizer
        train_dataloader: Training dataloader
        val_dataloader: Optional validation dataloader
        scheduler: Noise scheduler for diffusion
    """
    
    def __init__(
        self,
        config: DiffusionTrainingConfig,
        transformer: WanTransformer3DModel,
        geo_adapter: GeometryAdapter,
        vggt: VGGT,
        wan_vae: AutoencoderKLWan,
        text_encoder: WanT5EncoderModel,
        tokenizer: AutoTokenizer,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        scheduler: Any = None,
    ):
        self.config = config
        self.transformer = transformer
        self.geo_adapter = geo_adapter
        self.vggt = vggt
        self.wan_vae = wan_vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.scheduler = scheduler
        
        # Freeze models that shouldn't be trained
        self._freeze_model(self.geo_adapter)
        self._freeze_model(self.vggt)
        self._freeze_model(self.wan_vae)
        self._freeze_model(self.text_encoder)
        
        # Initialize loss function
        self.loss_fn = DiffusionLoss(prediction_type='noise', loss_type='l2')
        
        # Initialize optimizer for transformer only
        self.optimizer = AdamW(
            self.transformer.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Initialize scheduler
        total_steps = config.num_epochs * len(train_dataloader)
        self.lr_scheduler = self._get_scheduler(total_steps)
        
        # Initialize accelerator
        if ACCELERATE_AVAILABLE:
            self.accelerator = Accelerator(
                mixed_precision=config.mixed_precision,
                gradient_accumulation_steps=1,
            )
            (
                self.transformer,
                self.geo_adapter,
                self.vggt,
                self.wan_vae,
                self.text_encoder,
                self.optimizer,
                self.train_dataloader,
                self.lr_scheduler,
            ) = self.accelerator.prepare(
                self.transformer,
                self.geo_adapter,
                self.vggt,
                self.wan_vae,
                self.text_encoder,
                self.optimizer,
                self.train_dataloader,
                self.lr_scheduler,
            )
            self.device = self.accelerator.device
        else:
            self.accelerator = None
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.transformer = self.transformer.to(self.device)
            self.geo_adapter = self.geo_adapter.to(self.device)
            self.vggt = self.vggt.to(self.device)
            self.wan_vae = self.wan_vae.to(self.device)
            self.text_encoder = self.text_encoder.to(self.device)
        
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
    
    def _sample_task(self) -> str:
        """Sample a training task (1view, 2view, allview)."""
        r = random.random()
        if r < self.config.task_1view_prob:
            return "1view"
        elif r < self.config.task_1view_prob + self.config.task_2view_prob:
            return "2view"
        else:
            return "allview"
    
    def _should_drop_condition(self, prob: float) -> bool:
        """Determine whether to drop a condition."""
        return random.random() < prob
    
    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode images to get appearance and geometry latent codes.
        
        Args:
            images: Input images [B, F, 3, H, W] in [0, 1]
            
        Returns:
            Tuple of (appearance_latent, geometry_latent)
        """
        B, F, C, H, W = images.shape
        
        # Encode with Wan VAE for appearance latent A
        images_vae = rearrange((images * 2 - 1).clamp(-1, 1), 'b f c h w -> b c f h w')
        appearance_latent = self.wan_vae.encode(images_vae).latent_dist.sample()
        
        # Extract VGGT tokens and encode with geometry adapter for G
        aggregated_tokens_list, ps_idx = self.vggt.aggregator(images)
        intermediate_idx = self.vggt.depth_head.intermediate_layer_idx
        tokens = [aggregated_tokens_list[i] for i in intermediate_idx]
        concat_tokens = torch.cat(tokens, dim=-1)
        
        patch_size = self.vggt.aggregator.patch_size
        h_patches = H // patch_size
        w_patches = W // patch_size
        
        tokens_reshaped = concat_tokens[:, ps_idx:, :].reshape(B, F, h_patches, w_patches, -1)
        tokens_reshaped = rearrange(tokens_reshaped, 'b f h w c -> b c f h w')
        
        geometry_latent = self.geo_adapter.encode(tokens_reshaped).latent_dist.sample()
        
        return appearance_latent, geometry_latent
    
    @torch.no_grad()
    def encode_prompt(
        self, 
        prompts: List[str],
        drop_text: bool = False
    ) -> torch.Tensor:
        """
        Encode text prompts.
        
        Args:
            prompts: List of text prompts
            drop_text: Whether to drop text (use empty prompts)
            
        Returns:
            Prompt embeddings
        """
        if drop_text:
            prompts = [""] * len(prompts)
        
        # Tokenize
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        attention_mask = text_inputs.attention_mask.to(self.device)
        
        # Encode
        prompt_embeds = self.text_encoder(text_input_ids, attention_mask=attention_mask)[0]
        
        return prompt_embeds
    
    def prepare_condition_latents(
        self,
        images: torch.Tensor,
        task: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare conditioning latents based on task type.
        
        Args:
            images: Input images [B, F, 3, H, W]
            task: Task type ('1view', '2view', 'allview')
            
        Returns:
            Tuple of (condition_latents, mask)
        """
        B, F, C, H, W = images.shape
        
        # Get appearance latent shape
        appearance_latent, geometry_latent = self.encode_images(images)
        _, latent_c, latent_f, latent_h, latent_w = appearance_latent.shape
        
        # Create condition images based on task
        if task == "1view":
            # Only first frame
            cond_images = torch.zeros_like(images)
            cond_images[:, 0] = images[:, 0]
            mask = torch.zeros(B, F, latent_h, latent_w * 2, device=self.device)
            mask[:, 0, :, :latent_w] = 1  # Appearance only for first frame
            
        elif task == "2view":
            # First and last frames
            cond_images = torch.zeros_like(images)
            cond_images[:, 0] = images[:, 0]
            cond_images[:, -1] = images[:, -1]
            mask = torch.zeros(B, F, latent_h, latent_w * 2, device=self.device)
            mask[:, 0, :, :latent_w] = 1
            mask[:, -1, :, :latent_w] = 1
            
        else:  # allview
            cond_images = images
            mask = torch.ones(B, F, latent_h, latent_w * 2, device=self.device)
            mask[:, :, :, latent_w:] = 0  # No geometry conditioning
        
        # Encode condition images
        cond_app_latent, _ = self.encode_images(cond_images)
        
        # Geometry condition is zeros
        cond_geo_latent = torch.zeros_like(cond_app_latent)
        
        # Concatenate: Z_cond = [A_cond; G_cond]
        cond_latent = torch.cat([cond_app_latent, cond_geo_latent], dim=-1)
        
        return cond_latent, mask
    
    def add_noise(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to latents according to the scheduler."""
        if self.scheduler is not None:
            noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        else:
            # Simple linear schedule if no scheduler provided
            sqrt_alpha = (1 - timesteps.float() / 1000).sqrt().view(-1, 1, 1, 1, 1)
            sqrt_one_minus_alpha = (timesteps.float() / 1000).sqrt().view(-1, 1, 1, 1, 1)
            noisy_latents = sqrt_alpha * latents + sqrt_one_minus_alpha * noise
        return noisy_latents
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            batch: Batch of training data
            
        Returns:
            Dictionary of loss values
        """
        self.transformer.train()
        
        # Get input images and prompts
        images = batch["frames"].to(self.device)  # [B, F, 3, H, W]
        prompts = batch.get("prompt", [""] * images.shape[0])
        
        B = images.shape[0]
        
        # Sample task
        task = self._sample_task()
        
        # Determine condition dropout
        drop_text = self._should_drop_condition(self.config.text_dropout_prob)
        drop_camera = self._should_drop_condition(self.config.camera_dropout_prob)
        
        # Encode images to get joint latent Z = [A; G]
        appearance_latent, geometry_latent = self.encode_images(images)
        joint_latent = torch.cat([appearance_latent, geometry_latent], dim=-1)  # [B, C, f, h, 2w]
        
        # Prepare conditioning
        cond_latent, mask = self.prepare_condition_latents(images, task)
        
        # Encode prompts
        prompt_embeds = self.encode_prompt(prompts, drop_text=drop_text)
        
        # Sample noise and timesteps
        noise = torch.randn_like(joint_latent)
        timesteps = torch.randint(
            0, 1000, (B,), device=self.device, dtype=torch.long
        )
        
        # Add noise to latents
        noisy_latents = self.add_noise(joint_latent, noise, timesteps)
        
        # Camera conditioning (plucker embeddings)
        if "extrinsics" in batch and "intrinsics" in batch and not drop_camera:
            camera_latents = self._compute_camera_latents(batch)
        else:
            # Zero camera conditioning
            camera_latents = torch.zeros(
                B, 24, noisy_latents.shape[2], images.shape[-2], images.shape[-1] * 2,
                device=self.device, dtype=noisy_latents.dtype
            )
        
        # Forward through transformer
        model_output = self.transformer(
            x=noisy_latents,
            context=prompt_embeds,
            t=timesteps,
            seq_len=self._compute_seq_len(noisy_latents),
            y=cond_latent,
            y_camera=camera_latents,
        )
        
        # Compute loss
        loss = self.loss_fn(model_output, noise)
        
        # Backward pass
        if self.accelerator is not None:
            self.accelerator.backward(loss)
        else:
            loss.backward()
        
        # Gradient clipping
        if self.config.max_grad_norm > 0:
            if self.accelerator is not None:
                self.accelerator.clip_grad_norm_(
                    self.transformer.parameters(),
                    self.config.max_grad_norm
                )
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.transformer.parameters(),
                    self.config.max_grad_norm
                )
        
        # Optimizer step
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        
        return {
            "loss": loss.item(),
            "task": task,
        }
    
    def _compute_camera_latents(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute camera conditioning from extrinsics and intrinsics."""
        from ..utils.data_utils import compute_rays
        
        extrinsics = batch["extrinsics"].to(self.device)  # [B, F, 4, 4]
        intrinsics = batch["intrinsics"].to(self.device)  # [B, F, 3, 3]
        
        B, F, _, _ = extrinsics.shape
        H, W = self.config.resolution, self.config.resolution
        
        # Compute c2w (camera to world)
        c2ws = torch.linalg.inv(extrinsics)
        
        plucker_list = []
        for b in range(B):
            rays_o, rays_d = compute_rays(c2ws[b], intrinsics[b], H, W, self.device)
            o_cross_d = torch.cross(rays_o, rays_d, dim=1)
            plucker = torch.cat([o_cross_d, rays_d], dim=1)  # [F, 6, H, W]
            plucker_list.append(plucker)
        
        plucker_embeddings = torch.stack(plucker_list, dim=0)  # [B, F, 6, H, W]
        
        # Process for transformer input
        plucker_embeddings = plucker_embeddings.transpose(1, 2)  # [B, 6, F, H, W]
        plucker_embeddings = torch.cat([
            torch.repeat_interleave(plucker_embeddings[:, :, 0:1], repeats=4, dim=2),
            plucker_embeddings[:, :, 1:]
        ], dim=2).transpose(1, 2)  # [B, F+3, 6, H, W]
        
        num_frames = plucker_embeddings.shape[1]
        plucker_embeddings = plucker_embeddings.view(
            B, (num_frames) // 4, 4, 6, H, W
        ).transpose(2, 3)  # [B, f, 6, 4, H, W]
        
        plucker_embeddings = plucker_embeddings.view(
            B, (num_frames) // 4, 24, H, W
        ).transpose(1, 2)  # [B, 24, f, H, W]
        
        # Duplicate for joint latent space
        plucker_embeddings = torch.cat([plucker_embeddings, plucker_embeddings], dim=-1)
        
        return plucker_embeddings
    
    def _compute_seq_len(self, latents: torch.Tensor) -> int:
        """Compute sequence length for transformer."""
        _, c, t, h, w = latents.shape
        patch_size = self.transformer.config.patch_size
        return math.ceil((h * w) / (patch_size[1] * patch_size[2]) * t)
    
    def save_checkpoint(self, path: Optional[str] = None) -> None:
        """Save training checkpoint."""
        if path is None:
            path = os.path.join(
                self.config.output_dir,
                f"checkpoint-{self.global_step}.pt"
            )
        
        if self.accelerator is not None:
            model_state = self.accelerator.unwrap_model(self.transformer).state_dict()
        else:
            model_state = self.transformer.state_dict()
        
        checkpoint = {
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.lr_scheduler.state_dict(),
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "config": self.config,
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        if self.accelerator is not None:
            self.accelerator.unwrap_model(self.transformer).load_state_dict(
                checkpoint["model_state_dict"]
            )
        else:
            self.transformer.load_state_dict(checkpoint["model_state_dict"])
        
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.current_epoch = checkpoint["current_epoch"]
        
        logger.info(f"Checkpoint loaded from {path}")
    
    def train(self) -> None:
        """Run the full training loop."""
        logger.info("Starting diffusion model training...")
        logger.info(f"Config: {self.config}")
        
        # Load checkpoint if specified
        if self.config.checkpoint_path is not None:
            self.load_checkpoint(self.config.checkpoint_path)
        
        # Training loop
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            epoch_loss = 0.0
            task_counts = {"1view": 0, "2view": 0, "allview": 0}
            
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}/{self.config.num_epochs}",
                disable=self.accelerator is not None and not self.accelerator.is_main_process,
            )
            
            for batch in progress_bar:
                # Training step
                result = self.train_step(batch)
                self.global_step += 1
                
                epoch_loss += result["loss"]
                task_counts[result["task"]] += 1
                
                # Update progress bar
                progress_bar.set_postfix({"loss": f"{result['loss']:.4f}"})
                
                # Logging
                if self.global_step % self.config.log_every == 0:
                    logger.info(
                        f"Step {self.global_step}: loss={result['loss']:.4f}, task={result['task']}"
                    )
                
                # Checkpointing
                if self.global_step % self.config.save_every == 0:
                    self.save_checkpoint()
            
            # End of epoch
            avg_loss = epoch_loss / len(self.train_dataloader)
            logger.info(
                f"Epoch {epoch + 1}: avg_loss={avg_loss:.4f}, tasks={task_counts}"
            )
        
        # Final checkpoint
        self.save_checkpoint(os.path.join(self.config.output_dir, "final_checkpoint.pt"))
        logger.info("Training completed!")


def train_diffusion_model(
    data_root: str,
    pretrained_path: str,
    output_dir: str = "./outputs/diffusion",
    **kwargs
) -> WanTransformer3DModel:
    """
    Convenience function to train the diffusion model.
    
    Args:
        data_root: Path to training data
        pretrained_path: Path to pre-trained Gen3R checkpoint
        output_dir: Output directory for checkpoints
        **kwargs: Additional training configuration
        
    Returns:
        Trained transformer model
    """
    from .datasets import get_dataloader
    from ..pipeline import Gen3RPipeline
    
    # Create config
    config = DiffusionTrainingConfig(output_dir=output_dir, **kwargs)
    
    # Load pre-trained pipeline
    pipeline = Gen3RPipeline.from_pretrained(pretrained_path)
    
    # Create dataloader
    train_dataloader = get_dataloader(
        data_root=data_root,
        split="train",
        batch_size=config.batch_size,
        num_frames=config.num_frames,
        resolution=(config.resolution, config.resolution),
    )
    
    # Create trainer
    trainer = DiffusionTrainer(
        config=config,
        transformer=pipeline.transformer,
        geo_adapter=pipeline.geo_adapter,
        vggt=pipeline.vggt,
        wan_vae=pipeline.wan_vae,
        text_encoder=pipeline.text_encoder,
        tokenizer=pipeline.tokenizer,
        train_dataloader=train_dataloader,
        scheduler=pipeline.scheduler,
    )
    
    # Train
    trainer.train()
    
    return pipeline.transformer
