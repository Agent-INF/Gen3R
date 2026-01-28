"""
Loss functions for Gen3R training.

Implements the losses described in the paper:
- Reconstruction Loss (L_rec): Ensures adapter can reconstruct VGGT tokens and 3D attributes
- Distribution Alignment Loss (L_KL): Aligns geometry latent distribution with appearance latent distribution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class GeometryAdapterLoss(nn.Module):
    """
    Combined loss for training the Geometry Adapter.
    
    The total loss is: L = λ1 * L_rec + λ2 * L_KL
    
    where:
    - L_rec: Reconstruction loss for VGGT tokens and 3D attributes
    - L_KL: KL divergence between geometry and appearance latent distributions
    
    Args:
        lambda_rec: Weight for reconstruction loss (default: 1.0)
        lambda_kl: Weight for KL divergence loss (default: 0.001)
        token_weight: Weight for token reconstruction (default: 1.0)
        depth_weight: Weight for depth reconstruction (default: 1.0)
        camera_weight: Weight for camera reconstruction (default: 1.0)
        pointmap_weight: Weight for point map reconstruction (default: 1.0)
    """
    
    def __init__(
        self,
        lambda_rec: float = 1.0,
        lambda_kl: float = 0.001,
        token_weight: float = 1.0,
        depth_weight: float = 1.0,
        camera_weight: float = 1.0,
        pointmap_weight: float = 1.0,
    ):
        super().__init__()
        self.lambda_rec = lambda_rec
        self.lambda_kl = lambda_kl
        self.token_weight = token_weight
        self.depth_weight = depth_weight
        self.camera_weight = camera_weight
        self.pointmap_weight = pointmap_weight
        
    def forward(
        self,
        reconstructed_tokens: torch.Tensor,
        original_tokens: torch.Tensor,
        geometry_latent: torch.Tensor,
        appearance_latent: torch.Tensor,
        reconstructed_depth: Optional[torch.Tensor] = None,
        original_depth: Optional[torch.Tensor] = None,
        reconstructed_cameras: Optional[torch.Tensor] = None,
        original_cameras: Optional[torch.Tensor] = None,
        reconstructed_pointmap: Optional[torch.Tensor] = None,
        original_pointmap: Optional[torch.Tensor] = None,
        geometry_mu: Optional[torch.Tensor] = None,
        geometry_logvar: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the combined training loss.
        
        Args:
            reconstructed_tokens: Reconstructed VGGT tokens [B, C, T, H, W]
            original_tokens: Original VGGT tokens [B, C, T, H, W]
            geometry_latent: Geometry latent codes [B, C, T, H, W]
            appearance_latent: Appearance latent codes from Wan VAE [B, C, T, H, W]
            reconstructed_depth: Reconstructed depth maps (optional)
            original_depth: Original/GT depth maps (optional)
            reconstructed_cameras: Reconstructed camera parameters (optional)
            original_cameras: Original camera parameters (optional)
            reconstructed_pointmap: Reconstructed point maps (optional)
            original_pointmap: Original point maps (optional)
            geometry_mu: Mean of geometry latent distribution (optional)
            geometry_logvar: Log variance of geometry latent distribution (optional)
            
        Returns:
            Dictionary containing individual losses and total loss
        """
        losses = {}
        
        # Token reconstruction loss (L2)
        token_loss = reconstruction_loss(
            reconstructed_tokens, 
            original_tokens, 
            loss_type='l2'
        )
        losses['token_loss'] = token_loss * self.token_weight
        
        # Depth reconstruction loss (L2)
        if reconstructed_depth is not None and original_depth is not None:
            depth_loss = reconstruction_loss(
                reconstructed_depth,
                original_depth,
                loss_type='l2'
            )
            losses['depth_loss'] = depth_loss * self.depth_weight
        else:
            losses['depth_loss'] = torch.tensor(0.0, device=reconstructed_tokens.device)
        
        # Camera reconstruction loss (L2)
        if reconstructed_cameras is not None and original_cameras is not None:
            camera_loss = reconstruction_loss(
                reconstructed_cameras,
                original_cameras,
                loss_type='l2'
            )
            losses['camera_loss'] = camera_loss * self.camera_weight
        else:
            losses['camera_loss'] = torch.tensor(0.0, device=reconstructed_tokens.device)
        
        # Point map reconstruction loss (L1)
        if reconstructed_pointmap is not None and original_pointmap is not None:
            pointmap_loss = reconstruction_loss(
                reconstructed_pointmap,
                original_pointmap,
                loss_type='l1'
            )
            losses['pointmap_loss'] = pointmap_loss * self.pointmap_weight
        else:
            losses['pointmap_loss'] = torch.tensor(0.0, device=reconstructed_tokens.device)
        
        # Total reconstruction loss
        rec_loss = (
            losses['token_loss'] + 
            losses['depth_loss'] + 
            losses['camera_loss'] + 
            losses['pointmap_loss']
        )
        losses['rec_loss'] = rec_loss
        
        # Distribution alignment loss (KL divergence)
        if geometry_mu is not None and geometry_logvar is not None:
            # KL divergence between geometry latent and appearance latent distribution
            kl_loss = distribution_alignment_loss(
                geometry_mu,
                geometry_logvar,
                appearance_latent
            )
        else:
            # Approximate KL from latents directly
            kl_loss = distribution_alignment_loss_from_samples(
                geometry_latent,
                appearance_latent
            )
        losses['kl_loss'] = kl_loss
        
        # Total loss
        total_loss = self.lambda_rec * rec_loss + self.lambda_kl * kl_loss
        losses['total_loss'] = total_loss
        
        return losses


def reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = 'l2',
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute reconstruction loss between prediction and target.
    
    Args:
        pred: Predicted tensor
        target: Target tensor
        loss_type: Type of loss ('l1', 'l2', or 'smooth_l1')
        reduction: Reduction method ('mean', 'sum', or 'none')
        
    Returns:
        Reconstruction loss tensor
    """
    if loss_type == 'l1':
        loss = F.l1_loss(pred, target, reduction=reduction)
    elif loss_type == 'l2':
        loss = F.mse_loss(pred, target, reduction=reduction)
    elif loss_type == 'smooth_l1':
        loss = F.smooth_l1_loss(pred, target, reduction=reduction)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    return loss


def distribution_alignment_loss(
    geometry_mu: torch.Tensor,
    geometry_logvar: torch.Tensor,
    appearance_latent: torch.Tensor,
) -> torch.Tensor:
    """
    Compute KL divergence between geometry latent distribution q_G and 
    appearance latent distribution q_A.
    
    As described in the paper, this aligns the geometry latent distribution
    with the pre-trained appearance VAE's latent distribution.
    
    Args:
        geometry_mu: Mean of geometry latent distribution [B, C, T, H, W]
        geometry_logvar: Log variance of geometry latent [B, C, T, H, W]
        appearance_latent: Sample from appearance latent distribution [B, C, T, H, W]
        
    Returns:
        KL divergence loss
    """
    # Estimate appearance distribution statistics from samples
    # Using batch statistics as proxy for the distribution
    app_mu = appearance_latent.mean(dim=0, keepdim=True)
    app_var = appearance_latent.var(dim=0, keepdim=True) + 1e-8
    app_logvar = torch.log(app_var)
    
    # KL divergence: KL(N(μ1, σ1²) || N(μ2, σ2²))
    # = 0.5 * (log(σ2²/σ1²) + (σ1² + (μ1-μ2)²)/σ2² - 1)
    geometry_var = torch.exp(geometry_logvar)
    
    kl = 0.5 * (
        app_logvar - geometry_logvar +
        (geometry_var + (geometry_mu - app_mu).pow(2)) / app_var - 1
    )
    
    return kl.mean()


def distribution_alignment_loss_from_samples(
    geometry_latent: torch.Tensor,
    appearance_latent: torch.Tensor,
) -> torch.Tensor:
    """
    Approximate KL divergence between geometry and appearance latent distributions
    using sample statistics.
    
    Args:
        geometry_latent: Samples from geometry latent [B, C, T, H, W]
        appearance_latent: Samples from appearance latent [B, C, T, H, W]
        
    Returns:
        Approximate KL divergence loss
    """
    # Compute batch statistics for both distributions
    geo_mu = geometry_latent.mean(dim=0)
    geo_var = geometry_latent.var(dim=0) + 1e-8
    
    app_mu = appearance_latent.mean(dim=0)
    app_var = appearance_latent.var(dim=0) + 1e-8
    
    # KL divergence using estimated statistics
    kl = 0.5 * (
        torch.log(app_var / geo_var) +
        (geo_var + (geo_mu - app_mu).pow(2)) / app_var - 1
    )
    
    return kl.mean()


class DiffusionLoss(nn.Module):
    """
    Loss function for diffusion model training.
    
    Standard diffusion loss that predicts noise (or velocity) from noised latents.
    
    Args:
        prediction_type: Type of prediction ('noise' or 'v_prediction')
        loss_type: Type of loss ('l2', 'l1', or 'smooth_l1')
    """
    
    def __init__(
        self,
        prediction_type: str = 'noise',
        loss_type: str = 'l2',
    ):
        super().__init__()
        self.prediction_type = prediction_type
        self.loss_type = loss_type
        
    def forward(
        self,
        model_output: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute diffusion loss.
        
        Args:
            model_output: Model's prediction (noise or velocity)
            target: Target (noise or velocity)
            mask: Optional mask for selective loss computation
            
        Returns:
            Diffusion loss tensor
        """
        loss = reconstruction_loss(model_output, target, self.loss_type, 'none')
        
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / (mask.sum() + 1e-8)
        else:
            loss = loss.mean()
            
        return loss
