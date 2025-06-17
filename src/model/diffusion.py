"""
DDPM Diffusion Model Implementation
Based on "Denoising Diffusion Probabilistic Models" by Ho et al.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, Tuple
from tqdm import tqdm


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
    """
    Extract coefficients from a based on t and reshape to broadcast with x_shape.
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def linear_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    """Linear beta schedule for diffusion"""
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine beta schedule for diffusion"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model
    """
    
    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        model_mean_type: str = "epsilon",
        model_var_type: str = "fixed_small",
        loss_type: str = "mse",
    ):
        super().__init__()
        
        self.model = model
        self.timesteps = timesteps
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        
        # Define beta schedule
        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
            
        self.register_buffer("betas", betas)
        
        # Pre-compute useful quantities
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        
        # Log calculation clipped because the posterior variance is 0 at the beginning
        self.register_buffer("posterior_log_variance_clipped", 
                           torch.log(torch.cat([posterior_variance[1:2], posterior_variance[1:]])))
        self.register_buffer("posterior_mean_coef1", 
                           betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2", 
                           (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))
        
    def q_mean_variance(self, x_start: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the distribution q(x_t | x_0).
        """
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance
        
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        
    def q_posterior_mean_variance(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the mean and variance of the diffusion posterior:
        q(x_{t-1} | x_t, x_0)
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
        
    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Predict x_0 from x_t and predicted noise.
        """
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
        
    def p_mean_variance(self, x: torch.Tensor, t: torch.Tensor, clip_denoised: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply the model to get p(x_{t-1} | x_t).
        """
        B, C = x.shape[:2]
        assert t.shape == (B,)
        
        model_output = self.model(x, t)
        
        if self.model_var_type in ["learned", "learned_range"]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = torch.split(model_output, C, dim=1)
            
            if self.model_var_type == "learned":
                model_log_variance = model_var_values
                model_variance = torch.exp(model_log_variance)
            else:
                min_log = extract(self.posterior_log_variance_clipped, t, x.shape)
                max_log = extract(torch.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var]
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = torch.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                "fixed_large": (
                    torch.cat([self.posterior_variance[1:2], self.betas[1:]]),
                    torch.log(torch.cat([self.posterior_variance[1:2], self.betas[1:]])),
                ),
                "fixed_small": (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            
            model_variance = extract(model_variance, t, x.shape)
            model_log_variance = extract(model_log_variance, t, x.shape)
            
        def process_x_start(x_start):
            if clip_denoised:
                return x_start.clamp(-1, 1)
            return x_start
            
        if self.model_mean_type == "epsilon":
            pred_x_start = process_x_start(self.predict_start_from_noise(x, t, model_output))
        elif self.model_mean_type == "x0":
            pred_x_start = process_x_start(model_output)
        else:
            raise ValueError(f"Unknown model_mean_type: {self.model_mean_type}")
            
        model_mean, _, _ = self.q_posterior_mean_variance(pred_x_start, x, t)
        
        return model_mean, model_variance, model_log_variance
        
    def p_sample(self, x: torch.Tensor, t: torch.Tensor, clip_denoised: bool = True) -> torch.Tensor:
        """
        Sample x_{t-1} from the model at the given timestep.
        """
        model_mean, _, model_log_variance = self.p_mean_variance(x, t, clip_denoised=clip_denoised)
        noise = torch.randn_like(x)
        # No noise when t == 0
        nonzero_mask = (t != 0).float().reshape(-1, *([1] * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        
    def p_sample_loop(self, shape: tuple, device: torch.device, progress: bool = True) -> torch.Tensor:
        """
        Generate samples from the model.
        """
        b = shape[0]
        img = torch.randn(shape, device=device)
        
        indices = list(range(self.timesteps))[::-1]
        
        if progress:
            indices = tqdm(indices, desc="Sampling")
            
        for i in indices:
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t)
            
        return img
        
    def sample(self, batch_size: int, image_size: int, channels: int = 3, device: torch.device = None) -> torch.Tensor:
        """
        Generate samples from the model.
        """
        if device is None:
            device = next(self.parameters()).device
            
        shape = (batch_size, channels, image_size, image_size)
        return self.p_sample_loop(shape, device)
        
    def training_losses(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute training losses for a batch of samples.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        x_t = self.q_sample(x_start, t, noise=noise)
        
        model_output = self.model(x_t, t)
        
        if self.model_var_type in ["learned", "learned_range"]:
            B, C = x_start.shape[:2]
            assert model_output.shape == (B, C * 2, *x_start.shape[2:])
            model_output, model_var_values = torch.split(model_output, C, dim=1)
            # Learn the variance using the variational bound, but don't let it affect our mean prediction
            frozen_out = torch.cat([model_output.detach(), model_var_values], dim=1)
            terms = self._vb_terms_bpd(x_start, x_t, t, frozen_out)
            if self.loss_type == "rescaled_mse":
                # Divide by 1000 for equivalence with initial implementation
                terms["loss"] = terms["mse"] / 1000.0
            else:
                terms["loss"] = terms["vb"]
            return terms
        else:
            if self.model_mean_type == "epsilon":
                target = noise
            elif self.model_mean_type == "x0":
                target = x_start
            else:
                raise ValueError(f"Unknown model_mean_type: {self.model_mean_type}")
                
            assert model_output.shape == target.shape == x_start.shape
            
            if self.loss_type == "mse":
                loss = F.mse_loss(model_output, target, reduction="none")
            elif self.loss_type == "l1":
                loss = F.l1_loss(model_output, target, reduction="none")
            elif self.loss_type == "huber":
                loss = F.smooth_l1_loss(model_output, target, reduction="none")
            else:
                raise ValueError(f"Unknown loss_type: {self.loss_type}")
                
            return loss.mean()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for training.
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()
        
        return self.training_losses(x, t)
