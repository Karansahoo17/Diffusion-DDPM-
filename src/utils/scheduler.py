import torch
import numpy as np
from typing import Union, Optional
import math


class NoiseScheduler:
    """Base class for noise scheduling in diffusion models."""
    
    def __init__(self, num_timesteps: int = 1000):
        self.num_timesteps = num_timesteps
        self.timesteps = torch.arange(num_timesteps)
    
    def get_betas(self) -> torch.Tensor:
        """Return beta schedule."""
        raise NotImplementedError
    
    def get_alphas(self) -> torch.Tensor:
        """Return alpha schedule."""
        betas = self.get_betas()
        return 1.0 - betas
    
    def get_alphas_cumprod(self) -> torch.Tensor:
        """Return cumulative product of alphas."""
        alphas = self.get_alphas()
        return torch.cumprod(alphas, dim=0)


class LinearScheduler(NoiseScheduler):
    """Linear noise schedule."""
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02
    ):
        super().__init__(num_timesteps)
        self.beta_start = beta_start
        self.beta_end = beta_end
    
    def get_betas(self) -> torch.Tensor:
        """Linear schedule from beta_start to beta_end."""
        return torch.linspace(
            self.beta_start,
            self.beta_end,
            self.num_timesteps,
            dtype=torch.float32
        )


class CosineScheduler(NoiseScheduler):
    """Cosine noise schedule (improved schedule from Nichol & Dhariwal)."""
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        s: float = 0.008,
        max_beta: float = 0.999
    ):
        super().__init__(num_timesteps)
        self.s = s
        self.max_beta = max_beta
    
    def get_alphas_cumprod(self) -> torch.Tensor:
        """Cosine schedule for alpha_cumprod."""
        steps = self.num_timesteps + 1
        x = torch.linspace(0, self.num_timesteps, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos(((x / self.num_timesteps) + self.s) / (1 + self.s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        return alphas_cumprod[1:]
    
    def get_betas(self) -> torch.Tensor:
        """Derive betas from alpha_cumprod."""
        alphas_cumprod = self.get_alphas_cumprod()
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
        betas = 1 - (alphas_cumprod / alphas_cumprod_prev)
        return torch.clamp(betas, 0, self.max_beta)


class QuadraticScheduler(NoiseScheduler):
    """Quadratic noise schedule."""
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02
    ):
        super().__init__(num_timesteps)
        self.beta_start = beta_start
        self.beta_end = beta_end
    
    def get_betas(self) -> torch.Tensor:
        """Quadratic schedule."""
        return torch.linspace(
            self.beta_start ** 0.5,
            self.beta_end ** 0.5,
            self.num_timesteps,
            dtype=torch.float32
        ) ** 2


class SigmoidScheduler(NoiseScheduler):
    """Sigmoid-based noise schedule."""
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        tau: float = 1.0
    ):
        super().__init__(num_timesteps)
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.tau = tau
    
    def get_betas(self) -> torch.Tensor:
        """Sigmoid schedule."""
        t = torch.linspace(-6, 6, self.num_timesteps, dtype=torch.float32)
        sigmoid_values = torch.sigmoid(t / self.tau)
        # Normalize to [beta_start, beta_end]
        betas = self.beta_start + (self.beta_end - self.beta_start) * sigmoid_values
        return betas


class DDPMScheduler:
    """
    DDPM scheduler that handles forward and reverse diffusion processes.
    """
    
    def __init__(
        self,
        noise_scheduler: NoiseScheduler,
        clip_sample: bool = True,
        clip_sample_range: float = 1.0
    ):
        self.noise_scheduler = noise_scheduler
        self.num_timesteps = noise_scheduler.num_timesteps
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
        
        # Precompute all needed values
        self.betas = noise_scheduler.get_betas()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
    
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Add noise to original samples according to the noise schedule.
        
        Args:
            original_
