import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union


class DDPMLoss(nn.Module):
    """Standard DDPM loss function (MSE loss on noise prediction)."""
    
    def __init__(self, loss_type: str = 'mse'):
        super().__init__()
        self.loss_type = loss_type.lower()
        
    def forward(
        self,
        predicted_noise: torch.Tensor,
        target_noise: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute loss between predicted and target noise.
        
        Args:
            predicted_noise: Predicted noise from the model
            target_noise: Ground truth noise
            mask: Optional mask to ignore certain regions
            
        Returns:
            Loss value
        """
        if self.loss_type == 'mse':
            loss = F.mse_loss(predicted_noise, target_noise, reduction='none')
        elif self.loss_type == 'l1':
            loss = F.l1_loss(predicted_noise, target_noise, reduction='none')
        elif self.loss_type == 'huber':
            loss = F.huber_loss(predicted_noise, target_noise, reduction='none')
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
        
        # Apply mask if provided
        if mask is not None:
            loss = loss * mask
        
        # Reduce loss
        return loss.mean()


class WeightedDDPMLoss(nn.Module):
    """DDPM loss with timestep weighting."""
    
    def __init__(
        self,
        loss_type: str = 'mse',
        weighting_scheme: str = 'uniform',
        max_timesteps: int = 1000
    ):
        super().__init__()
        self.loss_type = loss_type.lower()
        self.weighting_scheme = weighting_scheme
        self.max_timesteps = max_timesteps
        
        # Precompute weights
        self.register_buffer('weights', self._compute_weights())
    
    def _compute_weights(self) -> torch.Tensor:
        """Compute timestep weights based on weighting scheme."""
        t = torch.arange(self.max_timesteps, dtype=torch.float32)
        
        if self.weighting_scheme == 'uniform':
            weights = torch.ones_like(t)
        elif self.weighting_scheme == 'snr':
            # Simple SNR-based weighting (approximation)
            # Higher weights for more difficult timesteps
            weights = (t / self.max_timesteps) ** 0.5
        elif self.weighting_scheme == 'sqrt':
            weights = torch.sqrt(t + 1)
        elif self.weighting_scheme == 'linear':
            weights = t / self.max_timesteps
        else:
            weights = torch.ones_like(t)
            
        return weights
    
    def forward(
        self,
        predicted_noise: torch.Tensor,
        target_noise: torch.Tensor,
        timesteps: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute weighted loss.
        
        Args:
            predicted_noise: Predicted noise from the model
            target_noise: Ground truth noise
            timesteps: Timesteps for each sample in the batch
            mask: Optional mask to ignore certain regions
            
        Returns:
            Weighted loss value
        """
        # Compute base loss
        if self.loss_type == 'mse':
            loss = F.mse_loss(predicted_noise, target_noise, reduction='none')
        elif self.loss_type == 'l1':
            loss = F.l1_loss(predicted_noise, target_noise, reduction='none')
        elif self.loss_type == 'huber':
            loss = F.huber_loss(predicted_noise, target_noise, reduction='none')
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
        
        # Get weights for current timesteps
        batch_weights = self.weights[timesteps]  # [batch_size]
        
        # Reshape weights to match loss dimensions
        while batch_weights.dim() < loss.dim():
            batch_weights = batch_weights.unsqueeze(-1)
        
        # Apply weights
        loss = loss * batch_weights
        
        # Apply mask if provided
        if mask is not None:
            loss = loss * mask
        
        # Reduce loss
        return loss.mean()


class VLBLoss(nn.Module):
    """Variational Lower Bound loss for DDPM."""
    
    def __init__(self, max_timesteps: int = 1000):
        super().__init__()
        self.max_timesteps = max_timesteps
    
    def forward(
        self,
        predicted_noise: torch.Tensor,
        target_noise: torch.Tensor,
        timesteps: torch.Tensor,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        alphas_cumprod: torch.Tensor,
        betas: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute VLB loss.
        
        Args:
            predicted_noise: Predicted noise from the model
            target_noise: Ground truth noise
            timesteps: Timesteps for each sample
            x_0: Original clean images
            x_t: Noisy images at timestep t
            alphas_cumprod: Cumulative product of alphas
            betas: Beta schedule
            
        Returns:
            VLB loss value
        """
        batch_size = predicted_noise.shape[0]
        
        # Compute MSE loss (denoising term)
        mse_loss = F.mse_loss(predicted_noise, target_noise, reduction='none')
        mse_loss = mse_loss.view(batch_size, -1).mean(dim=1)
        
        # Compute VLB weights
        alpha_t = alphas_cumprod[timesteps]
        beta_t = betas[timesteps]
        
        # VLB weighting term
        vlb_weights = beta_t ** 2 / (2 * (1 - alpha_t) * (1 - alpha_t / (1 - beta_t)) ** 2)
        
        # Apply weights
        vlb_loss = mse_loss * vlb_weights
        
        return vlb_loss.mean()


class PerceptualLoss(nn.Module):
    """Perceptual loss using pre-trained VGG features."""
    
    def __init__(self, layers: list = [3, 8, 15, 22], weights: list = [1.0, 1.0, 1.0, 1.0]):
        super().__init__()
        
        # Load pre-trained VGG
        import torchvision.models as models
        vgg = models.vgg16(pretrained=True).features
        
        # Extract specified layers
        self.layers = nn.ModuleList()
        layer_idx = 0
        current_layer = nn.Sequential()
        
        for i, layer in enumerate(vgg):
            current_layer.add_module(str(i), layer)
            if i in layers:
                self.layers.append(current_layer)
                current_layer = nn.Sequential()
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
        
        self.weights = weights
    
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss.
        
        Args:
            predicted: Predicted images
            target: Target images
            
        Returns:
            Perceptual loss value
        """
        # Ensure inputs are in [0, 1] range and have 3 channels
        if predicted.shape[1] == 1:
            predicted = predicted.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        
        # Normalize for VGG (ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(predicted.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(predicted.device)
        
        predicted = (predicted - mean) / std
        target = (target - mean) / std
        
        # Extract features
        pred_features = []
        target_features = []
        
        pred_x = predicted
        target_x = target
        
        for layer in self.layers:
            pred_x = layer(pred_x)
            target_x = layer(target_x)
            pred_features.append(pred_x)
            target_features.append(target_x)
        
        # Compute loss
        loss = 0.0
        for i, (pred_feat, target_feat) in enumerate(zip(pred_features, target_features)):
            loss += self.weights[i] * F.mse_loss(pred_feat, target_feat)
        
        return loss


class CombinedLoss(nn.Module):
    """Combined loss function with multiple components."""
    
    def __init__(
        self,
        losses: dict,
        weights: dict = None
    ):
        """
        Initialize combined loss.
        
        Args:
            losses: Dictionary of loss functions
            weights: Dictionary of loss weights
        """
        super().__init__()
        self.losses = nn.ModuleDict(losses)
        self.weights = weights or {name: 1.0 for name in losses.keys()}
    
    def forward(self, *args, **kwargs) -> dict:
        """
        Compute combined loss.
        
        Returns:
            Dictionary with individual losses and total loss
        """
        loss_dict = {}
        total_loss = 0.0
        
        for name, loss_fn in self.losses.items():
            loss_value = loss_fn(*args, **kwargs)
            loss_dict[name] = loss_value
            total_loss += self.weights[name] * loss_value
        
        loss_dict['total'] = total_loss
        return loss_dict


# Factory functions
def create_ddpm_loss(loss_type: str = 'mse') -> DDPMLoss:
    """Create standard DDPM loss."""
    return DDPMLoss(loss_type=loss_type)


def create_weighted_loss(
    loss_type: str = 'mse',
    weighting_scheme: str = 'uniform',
    max_timesteps: int = 1000
) -> WeightedDDPMLoss:
    """Create weighted DDPM loss."""
    return WeightedDDPMLoss(
        loss_type=loss_type,
        weighting_scheme=weighting_scheme,
        max_timesteps=max_timesteps
    )


def create_combined_loss(
    base_loss: str = 'mse',
    use_perceptual: bool = False,
    perceptual_weight: float = 1.0
) -> Union[DDPMLoss, CombinedLoss]:
    """Create combined loss function."""
    if not use_perceptual:
        return create_ddpm_loss(base_loss)
    
    losses = {
        'mse': create_ddpm_loss(base_loss),
        'perceptual': PerceptualLoss()
    }
    
    weights = {
        'mse': 1.0,
        'perceptual': perceptual_weight
    }
    
    return CombinedLoss(losses, weights)
